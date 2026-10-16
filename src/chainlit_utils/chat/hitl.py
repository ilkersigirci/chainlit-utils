"""Event-driven OpenAI Responses HITL workflows for Chainlit."""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, Self

import chainlit as cl
from openai.types.responses import Response, ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from chainlit_utils.chat.history import (
    is_model_context,
    mark_model_context_excluded,
    send_ui_message,
)
from chainlit_utils.openai.responses import raise_for_response, response_input
from chainlit_utils.openai.tools import function_call_output, function_calls

logger = logging.getLogger(__name__)

HITL_LEDGER_METADATA_KEY = "chainlit_utils.hitl_ledger"
HITL_CONTROL_PROP = "_chainlit_utils_hitl"


class InvalidHitlSubmissionError(ValueError):
    """A browser submission does not match the current durable HITL state."""


class HitlLedger(BaseModel):
    """One review's paused function calls and, once answered, their outputs.

    The review message persists it, so a reload resumes the paused Response
    and later requests replay the answered review.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3] = 3
    model_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    function_calls: tuple[ResponseFunctionToolCall, ...] = Field(min_length=1)
    # None until the review is answered.
    outputs: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _complete_batch(self) -> Self:
        if len({call.call_id for call in self.function_calls}) != len(
            self.function_calls
        ):
            raise ValueError("HITL call IDs must be unique.")
        if self.outputs is not None:
            self.output_items(self.outputs)
        return self

    def call_items(self) -> list[dict[str, Any]]:
        """Return the function calls as Responses input items."""
        return [
            call.model_dump(mode="json", by_alias=True, exclude_none=True)
            for call in self.function_calls
        ]

    def output_items(self, outputs: Sequence[str]) -> list[dict[str, Any]]:
        """Return the function outputs that answer every call, in call order."""
        if len(outputs) != len(self.function_calls):
            raise ValueError("Every HITL function call requires one output.")
        return [
            function_call_output(call, output)
            for call, output in zip(self.function_calls, outputs, strict=True)
        ]


class _Submission(BaseModel):
    """The untrusted action payload; references echo ``HITL_CONTROL_PROP``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    step_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class PendingHitl:
    """A review awaiting answers: its Chainlit message and durable ledger."""

    message: cl.Message
    ledger: HitlLedger


class _ContinueResponse(Protocol):
    async def __call__(
        self,
        input_items: list[dict[str, Any]],
        *,
        model_id: str,
        previous_response_id: str,
    ) -> Response: ...


_PromptForCalls = Callable[[Sequence[ResponseFunctionToolCall]], str]
_PublishFinal = Callable[[Response], Awaitable[None]]
_ReviewProps = Callable[
    [Sequence[ResponseFunctionToolCall]],
    Mapping[str, object],
]
_ValidateOutputs = Callable[
    [Sequence[ResponseFunctionToolCall], Sequence[str]],
    Sequence[str],
]
_Receipt = Callable[[Sequence[str]], str]


class HitlWorkflow:
    """Publish durable controls and process one continuation per UI action."""

    def __init__(
        self,
        tool_name: str,
        *,
        action_name: str,
        continue_response: _ContinueResponse,
        element_name: str,
        prompt: _PromptForCalls,
        publish_final: _PublishFinal,
        review: _ReviewProps,
        validate_outputs: _ValidateOutputs | None = None,
        receipt: _Receipt | None = None,
    ) -> None:
        self._tool_name = tool_name
        self._action_name = action_name
        self._continue_response = continue_response
        self._element_name = element_name
        self._prompt = prompt
        self._publish_final = publish_final
        self._review = review
        self._validate_outputs = validate_outputs
        self._receipt = receipt

    async def publish(self, response: Response, *, model_id: str) -> PendingHitl | None:
        """Publish a terminal response or ask for review in a new message."""
        raise_for_response(response)
        calls = function_calls(response)
        # Only this workflow's tool pauses for review. Any other response,
        # including client function calls, belongs to publish_final.
        if any(call.name == self._tool_name for call in calls):
            return await self._send_review(response, calls, model_id=model_id)
        await self._publish_final(response)
        return None

    def response_input(self) -> list[dict[str, Any]]:
        """Return the chat as Responses input that replays answered reviews.

        Like a stateless Responses transcript, each answered review replays its
        function calls and outputs in place; other messages follow
        ``text_only_chat_messages``. A review records its answers before its
        continuation runs, so a transcript built meanwhile includes it.
        """
        items: list[dict[str, Any]] = []
        for message, openai_message in zip(
            cl.chat_context.get(),
            cl.chat_context.to_openai(),
            strict=True,
        ):
            ledger = _ledger(message)
            if ledger is None:
                if is_model_context(message):
                    items += response_input([openai_message])
            elif ledger.outputs is not None:
                items += ledger.call_items() + ledger.output_items(ledger.outputs)
        return items

    async def block_new_message(self, trigger_message: cl.Message) -> bool:
        """Reject a new user turn while the current thread awaits human input."""
        if self._current_pending() is None:
            return False
        mark_model_context_excluded(trigger_message)
        await trigger_message.update()
        await send_ui_message(
            "Resolve the pending interrupt before starting another request."
        )
        return True

    async def submit_action(self, action: cl.Action) -> dict[str, object]:
        """Record a review element's answers and return its ``callAction`` reply.

        Register this for the workflow's action name. The reply is
        ``{"ok": True}`` once the answers are recorded, or ``{"ok": False,
        "error": message}``, which the element shows. The continuation then runs
        as the session's task, like a chat message: Chainlit shows it running
        and Stop cancels it. A failed continuation is logged and reported in the
        chat.
        """
        try:
            submission = _Submission.model_validate(action.payload)
        except ValidationError:
            return {"ok": False, "error": "Invalid human-review submission."}
        try:
            ledger, input_items = await self._record(**submission.model_dump())
        except InvalidHitlSubmissionError as exc:
            return {"ok": False, "error": str(exc)}
        # Chainlit runs a chat message the same way; Stop cancels this task.
        cl.context.session.current_task = asyncio.create_task(
            self._run(ledger, input_items)
        )
        return {"ok": True}

    async def submit(
        self,
        *,
        step_id: str,
        element_id: str,
        revision: str,
        outputs: Sequence[str],
    ) -> PendingHitl | None:
        """Record a UI action's answers, then continue the paused Response."""
        ledger, input_items = await self._record(
            step_id=step_id,
            element_id=element_id,
            revision=revision,
            outputs=outputs,
        )
        return await self._continue(ledger, input_items)

    def _current_pending(self) -> PendingHitl | None:
        for message in reversed(cl.chat_context.get()):
            ledger = _ledger(message)
            if ledger is not None:
                return PendingHitl(message, ledger) if ledger.outputs is None else None
        return None

    async def _record(
        self,
        *,
        step_id: str,
        element_id: str,
        revision: str,
        outputs: Sequence[str],
    ) -> tuple[HitlLedger, list[dict[str, Any]]]:
        # Nothing awaits before the answers are recorded, so a duplicate
        # submission finds the review answered.
        pending = self._current_pending()
        if pending is None:
            raise InvalidHitlSubmissionError(
                "This human-review request is no longer pending."
            )
        message, ledger = pending.message, pending.ledger
        if (message.id, ledger.element_id, ledger.response_id) != (
            step_id,
            element_id,
            revision,
        ):
            raise InvalidHitlSubmissionError("This human-review request is stale.")
        try:
            answers = tuple(
                self._validate_outputs(ledger.function_calls, outputs)
                if self._validate_outputs is not None
                else outputs
            )
            input_items = ledger.output_items(answers)
        except (TypeError, ValueError) as exc:
            raise InvalidHitlSubmissionError(str(exc)) from exc

        message.elements = []
        if self._receipt is not None:
            # The answered review keeps its request and records the decision.
            message.content = f"{message.content}\n\n{self._receipt(answers)}"
        _set_ledger(message, ledger.model_copy(update={"outputs": answers}))
        _normalize_chainlit_created_at(message)
        await message.update()
        # A restored thread's message has no elements, so the control is
        # rebuilt from the ledger; removal needs only its ID.
        control = cl.CustomElement(id=ledger.element_id, name=self._element_name)
        try:
            await control.remove()
        except Exception:
            logger.exception("Failed to remove a completed HITL element")
        return ledger, input_items

    async def _continue(
        self, ledger: HitlLedger, input_items: list[dict[str, Any]]
    ) -> PendingHitl | None:
        response = await self._continue_response(
            input_items,
            model_id=ledger.model_id,
            previous_response_id=ledger.response_id,
        )
        return await self.publish(response, model_id=ledger.model_id)

    async def _run(self, ledger: HitlLedger, input_items: list[dict[str, Any]]) -> None:
        emitter = cl.context.emitter
        await emitter.task_start()
        try:
            await self._continue(ledger, input_items)
        except Exception as exc:
            logger.exception("Chainlit HITL continuation failed")
            await send_ui_message(f"Response failed: {exc}")
        finally:
            await emitter.task_end()

    async def _send_review(
        self,
        response: Response,
        calls: Sequence[ResponseFunctionToolCall],
        *,
        model_id: str,
    ) -> PendingHitl:
        if any(call.name != self._tool_name for call in calls):
            raise ValueError("A human review cannot include other function calls.")
        ledger = HitlLedger(
            model_id=model_id,
            response_id=response.id,
            element_id=str(uuid.uuid4()),
            function_calls=tuple(calls),
        )
        # The control's props name its message, so both IDs come first.
        message_id = str(uuid.uuid4())
        control = {
            "action": self._action_name,
            "element_id": ledger.element_id,
            "revision": ledger.response_id,
            "step_id": message_id,
        }
        # Like a chat transcript, every pause asks in a new message below the
        # answered reviews.
        message = cl.Message(
            id=message_id,
            content=self._prompt(calls),
            elements=[
                cl.CustomElement(
                    id=ledger.element_id,
                    name=self._element_name,
                    props={**self._review(calls), HITL_CONTROL_PROP: control},
                )
            ],
        )
        _set_ledger(message, ledger)
        await message.send()
        return PendingHitl(message, ledger)


def _ledger(message: cl.Message) -> HitlLedger | None:
    raw = (message.metadata or {}).get(HITL_LEDGER_METADATA_KEY)
    return None if raw is None else HitlLedger.model_validate(raw)


def _set_ledger(message: cl.Message, ledger: HitlLedger) -> None:
    # The review's text stays out of model requests; its calls carry it.
    mark_model_context_excluded(message)
    message.metadata = {
        **(message.metadata or {}),
        HITL_LEDGER_METADATA_KEY: ledger.model_dump(
            mode="json", by_alias=True, exclude_none=True
        ),
    }


def _normalize_chainlit_created_at(message: cl.Message) -> None:
    """Make an official-data-layer timestamp safe for a later message update."""
    if not message.created_at:
        return
    try:
        timestamp = datetime.fromisoformat(message.created_at)
    except ValueError:
        return
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    # Chainlit 2.12 serializes PostgreSQL timestamps without ``Z`` when a
    # thread is hydrated, but its update path parses only this exact form.
    message.created_at = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


__all__ = [
    "HITL_CONTROL_PROP",
    "HITL_LEDGER_METADATA_KEY",
    "HitlLedger",
    "HitlWorkflow",
    "InvalidHitlSubmissionError",
    "PendingHitl",
]
