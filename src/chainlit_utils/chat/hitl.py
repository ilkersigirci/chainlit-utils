"""Event-driven OpenAI Responses HITL workflows for Chainlit."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, cast

import chainlit as cl
from openai.types.responses import Response, ResponseFunctionToolCall

from chainlit_utils.chat.history import mark_model_context_excluded, send_ui_message
from chainlit_utils.openai.hitl import (
    HitlContinuation,
    HitlLedgerCodec,
    InvalidHitlLedgerError,
    function_call_outputs,
)
from chainlit_utils.openai.responses import raise_for_response
from chainlit_utils.openai.tools import function_calls

logger = logging.getLogger(__name__)

HITL_LEDGER_METADATA_KEY = "chainlit_utils.hitl_ledger"
HITL_ELEMENT_METADATA_KEY = "chainlit_utils.hitl_element_id"
HITL_CONTROL_PROP = "_chainlit_utils_hitl"


class InvalidHitlSubmissionError(ValueError):
    """A browser submission does not match the current durable HITL state."""


@dataclass(frozen=True, slots=True)
class PendingHitl:
    """A durable Responses continuation paired with its Chainlit message."""

    message: cl.Message
    continuation: HitlContinuation
    element_id: str


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class _SubmissionLocks:
    """Serialize submissions for one ledger step without retaining idle locks."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._entries: dict[str, _LockEntry] = {}

    @asynccontextmanager
    async def hold(self, step_id: str) -> AsyncIterator[None]:
        async with self._guard:
            entry = self._entries.setdefault(step_id, _LockEntry())
            entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            async with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(step_id, None)


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
    ) -> None:
        if not action_name:
            raise ValueError("HITL action name must be a non-empty string.")
        if not element_name:
            raise ValueError("HITL element name must be a non-empty string.")
        self._codec = HitlLedgerCodec(tool_name)
        self._action_name = action_name
        self._continue_response = continue_response
        self._element_name = element_name
        self._prompt = prompt
        self._publish_final = publish_final
        self._review = review
        self._validate_outputs = validate_outputs
        self._submission_locks = _SubmissionLocks()

    async def publish(self, response: Response, *, model_id: str) -> PendingHitl | None:
        """Publish a terminal response or persist its interrupt controls and return."""
        raise_for_response(response)
        return await self._transition(response, model_id=model_id)

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

    async def submit(
        self,
        *,
        step_id: str,
        element_id: str,
        revision: str,
        outputs: Sequence[str],
    ) -> PendingHitl | None:
        """Validate a UI action and advance one durable continuation."""
        if not all(
            isinstance(value, str) and value
            for value in (step_id, element_id, revision)
        ):
            raise InvalidHitlSubmissionError("The human-review reference is invalid.")
        if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
            raise InvalidHitlSubmissionError(
                "Human-review outputs must be a list of strings."
            )

        async with self._submission_locks.hold(step_id):
            pending = self._current_pending()
            if pending is None:
                raise InvalidHitlSubmissionError(
                    "This human-review request is no longer pending."
                )
            if pending.message.id != step_id:
                raise InvalidHitlSubmissionError("This human-review request is stale.")
            if pending.continuation.response_id != revision:
                raise InvalidHitlSubmissionError("This human-review revision is stale.")
            if pending.element_id != element_id:
                raise InvalidHitlSubmissionError(
                    "This human-review control is invalid or stale."
                )

            element = self._control_element(pending)
            validated_outputs = self._validated_outputs(
                pending.continuation.function_calls,
                outputs,
            )
            try:
                input_items = function_call_outputs(
                    pending.continuation,
                    validated_outputs,
                )
            except (TypeError, ValueError) as exc:
                raise InvalidHitlSubmissionError(str(exc)) from exc

            response = await self._continue_response(
                input_items,
                model_id=pending.continuation.model_id,
                previous_response_id=pending.continuation.response_id,
            )
            # A failed model response must leave the prior durable request intact.
            raise_for_response(response)
            return await self._transition(
                response,
                model_id=pending.continuation.model_id,
                ledger_message=pending.message,
                element=element,
            )

    def _current_pending(self) -> PendingHitl | None:
        for message in reversed(cl.chat_context.get()):
            metadata = message.metadata
            if not isinstance(metadata, Mapping):
                continue
            if HITL_LEDGER_METADATA_KEY not in metadata:
                continue
            continuation = self._codec.decode(metadata[HITL_LEDGER_METADATA_KEY])
            if continuation is None:
                return None
            if not isinstance(message, cl.Message):
                raise InvalidHitlLedgerError(
                    "The pending HITL step is not a Chainlit message."
                )
            element_id = metadata.get(HITL_ELEMENT_METADATA_KEY)
            if not isinstance(element_id, str) or not element_id:
                raise InvalidHitlLedgerError(
                    "The pending HITL element reference is invalid."
                )
            return PendingHitl(
                message=message,
                continuation=continuation,
                element_id=element_id,
            )
        return None

    def _validated_outputs(
        self,
        calls: Sequence[ResponseFunctionToolCall],
        outputs: Sequence[str],
    ) -> tuple[str, ...]:
        try:
            validated = (
                self._validate_outputs(calls, outputs)
                if self._validate_outputs is not None
                else outputs
            )
            if isinstance(validated, (str, bytes)) or not isinstance(
                validated, Sequence
            ):
                raise TypeError("Human-review outputs must be a list of strings.")
            if any(not isinstance(output, str) for output in validated):
                raise TypeError("Human-review outputs must be strings.")
            return tuple(validated)
        except (TypeError, ValueError) as exc:
            raise InvalidHitlSubmissionError(str(exc)) from exc

    async def _transition(
        self,
        response: Response,
        *,
        model_id: str,
        ledger_message: cl.Message | None = None,
        element: cl.CustomElement | None = None,
    ) -> PendingHitl | None:
        calls = function_calls(response)
        if calls:
            continuation = self._codec.continuation(
                model_id=model_id,
                response_id=response.id,
                function_calls=calls,
            )
            prompt = self._prompt(calls)
            message = ledger_message or cl.Message(content=prompt)
            control = element or cl.CustomElement(
                name=self._element_name,
                display="inline",
                props={},
            )
            props = dict(self._review(calls))
            props[HITL_CONTROL_PROP] = {
                "action": self._action_name,
                "element_id": control.id,
                "revision": continuation.response_id,
                "step_id": message.id,
            }
            content = json.dumps(props, ensure_ascii=False)
            control.props = props
            # Chainlit serializes CustomElement props only during construction.
            # Refresh content when an existing persisted element is reused.
            control.content = content
            message.content = prompt
            message.elements = cast("list[Any]", [control])
            _set_ledger_metadata(
                message,
                self._codec.pending_metadata(continuation),
                metadata_key=HITL_LEDGER_METADATA_KEY,
            )
            message.metadata = {
                **(message.metadata or {}),
                HITL_ELEMENT_METADATA_KEY: control.id,
            }
            _normalize_chainlit_created_at(message)
            if ledger_message is None:
                await message.send()
            else:
                await message.update()
            return PendingHitl(
                message=message,
                continuation=continuation,
                element_id=control.id,
            )

        if ledger_message is not None:
            ledger_message.elements = []
            _set_ledger_metadata(
                ledger_message,
                self._codec.completed_metadata(),
                metadata_key=HITL_LEDGER_METADATA_KEY,
            )
            ledger_message.metadata = {
                key: value
                for key, value in (ledger_message.metadata or {}).items()
                if key != HITL_ELEMENT_METADATA_KEY
            }
            # Persist completion before rendering final output so a reconnect
            # cannot replay the already-consumed continuation.
            _normalize_chainlit_created_at(ledger_message)
            await ledger_message.update()
            if element is not None:
                try:
                    await element.remove()
                except Exception:
                    logger.exception("Failed to remove a completed HITL element")
        await self._publish_final(response)
        return None

    def _control_element(self, pending: PendingHitl) -> cl.CustomElement:
        """Reuse the live control or reconstruct its persisted Chainlit identity."""
        for element in pending.message.elements:
            if (
                isinstance(element, cl.CustomElement)
                and element.id == pending.element_id
                and element.name == self._element_name
                and element.for_id == pending.message.id
            ):
                return cast(cl.CustomElement, element)
        return cl.CustomElement(
            thread_id=pending.message.thread_id,
            name=self._element_name,
            id=pending.element_id,
            display="inline",
            for_id=pending.message.id,
            props={},
        )


def _set_ledger_metadata(
    message: cl.Message,
    ledger: Mapping[str, object],
    *,
    metadata_key: str,
) -> None:
    mark_model_context_excluded(message)
    message.metadata = {
        **(message.metadata or {}),
        metadata_key: dict(ledger),
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
    "HITL_ELEMENT_METADATA_KEY",
    "HITL_LEDGER_METADATA_KEY",
    "HitlWorkflow",
    "InvalidHitlSubmissionError",
    "PendingHitl",
]
