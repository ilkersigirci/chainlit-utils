"""Run durable OpenAI Responses HITL workflows in Chainlit."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, cast

import chainlit as cl
from chainlit.context import context as chainlit_context
from chainlit.step import StepDict
from chainlit.types import ThreadDict
from openai.types.responses import Response, ResponseFunctionToolCall

from chainlit_utils.chat.history import (
    mark_model_context_excluded,
    mark_persisted_errors_excluded,
    send_ui_message,
)
from chainlit_utils.chat.resume import schedule_after_thread_hydration
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
PENDING_HITL_SESSION_KEY = "chainlit_utils.pending_hitl"
_LIVE_HITL_STATE_ATTRIBUTE = "_chainlit_utils_live_hitl_state"


@dataclass(frozen=True, slots=True)
class PendingHitl:
    """A durable Responses continuation paired with its Chainlit message."""

    message: cl.Message
    continuation: HitlContinuation


@dataclass(frozen=True, slots=True)
class _PendingLedgerEntry:
    """The host step and decoded value for the newest pending ledger."""

    step: Mapping[str, object]
    continuation: HitlContinuation


@dataclass(slots=True)
class _LiveHitlState:
    """Ephemeral HITL state owned by one live Chainlit WebSocket session."""

    disconnected_socket_id: str | None = None
    prompt_task: asyncio.Task[Any] | None = None


class _ContinueResponse(Protocol):
    async def __call__(
        self,
        input_items: list[dict[str, Any]],
        *,
        model_id: str,
        previous_response_id: str,
    ) -> Response: ...


class _PublishResponse(Protocol):
    async def __call__(
        self,
        response: Response,
        *,
        model_id: str,
        ledger_message: cl.Message,
    ) -> PendingHitl | None: ...


_AskForOutput = Callable[
    [ResponseFunctionToolCall, cl.Message],
    Awaitable[str | None],
]
_PromptForCalls = Callable[[Sequence[ResponseFunctionToolCall]], str]
_PublishFinal = Callable[[Response], Awaitable[None]]


class HitlWorkflow:
    """Coordinate one durable Responses HITL tool in a Chainlit application."""

    def __init__(
        self,
        tool_name: str,
        *,
        ask: _AskForOutput,
        continue_response: _ContinueResponse,
        prompt: _PromptForCalls,
        publish_final: _PublishFinal,
        element_name: str | None = None,
    ) -> None:
        self._codec = HitlLedgerCodec(tool_name)
        self._ask = ask
        self._continue_response = continue_response
        self._prompt = prompt
        self._publish_final = publish_final
        self._element_name = element_name

    async def run(self, response: Response, *, model_id: str) -> None:
        """Persist and resolve an interrupt Response through terminal output."""
        pending = await self._publish_response(response, model_id=model_id)
        if pending is not None:
            await self._resolve(pending)

    async def continue_pending(self, trigger_message: cl.Message | None = None) -> bool:
        """Resume the session's pending workflow instead of starting a new run."""
        pending = cl.user_session.get(PENDING_HITL_SESSION_KEY)
        if not isinstance(pending, PendingHitl):
            return False
        self._mark_connected()
        if trigger_message is not None:
            mark_model_context_excluded(trigger_message)
            await trigger_message.update()
        await send_ui_message(
            "Resolve the pending interrupt before starting another request."
        )
        await self._resolve(pending)
        return True

    async def restore(self, thread: ThreadDict) -> None:
        """Restore and reopen the newest pending workflow after thread hydration."""
        mark_persisted_errors_excluded(thread)
        self._mark_connected()
        self._live_state().prompt_task = None
        cl.user_session.set(PENDING_HITL_SESSION_KEY, None)
        try:
            pending = restore_pending_hitl(thread, codec=self._codec)
            if pending is None:
                return
            if self._element_name is not None:
                await remove_persisted_custom_elements(
                    thread,
                    step_id=pending.message.id,
                    element_name=self._element_name,
                )
            cl.user_session.set(PENDING_HITL_SESSION_KEY, pending)
            schedule_after_thread_hydration(partial(self._reopen, pending))
        except InvalidHitlLedgerError as exc:
            logger.exception("Persisted Chainlit HITL ledger is invalid")
            schedule_after_thread_hydration(
                partial(send_ui_message, f"Response failed: {exc}")
            )
        except Exception:
            logger.exception("Chainlit HITL resume failed")

    @staticmethod
    def cancel() -> None:
        """Record the disconnect and cancel only its live HITL prompt."""
        state = HitlWorkflow._live_state()
        socket_id = getattr(chainlit_context.session, "socket_id", None)
        if isinstance(socket_id, str):
            state.disconnected_socket_id = socket_id

        task = state.prompt_task
        if (
            isinstance(task, asyncio.Task)
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()

    async def _resolve(self, pending: PendingHitl) -> None:
        await resolve_hitl(
            pending,
            ask=self._ask_while_connected,
            continue_response=self._continue_response,
            publish_response=self._publish_response,
        )

    async def _ask_while_connected(
        self,
        call: ResponseFunctionToolCall,
        ledger_message: cl.Message,
    ) -> str | None:
        if self._is_disconnected():
            return None

        task = asyncio.current_task()
        if task is None:
            return await self._ask(call, ledger_message)

        state = self._live_state()
        state.prompt_task = task
        try:
            return await self._ask(call, ledger_message)
        finally:
            if state.prompt_task is task:
                state.prompt_task = None

    @staticmethod
    def _is_disconnected() -> bool:
        state = HitlWorkflow._live_state()
        if state.disconnected_socket_id is None:
            return False
        current_socket = getattr(chainlit_context.session, "socket_id", None)
        return state.disconnected_socket_id == current_socket

    @staticmethod
    def _mark_connected() -> None:
        HitlWorkflow._live_state().disconnected_socket_id = None

    @staticmethod
    def _live_state() -> _LiveHitlState:
        # Chainlit clears user_session on thread navigation while the old task can
        # still finish, so live ownership must remain on the referenced session.
        session = chainlit_context.session
        state = getattr(session, _LIVE_HITL_STATE_ATTRIBUTE, None)
        if not isinstance(state, _LiveHitlState):
            state = _LiveHitlState()
            setattr(session, _LIVE_HITL_STATE_ATTRIBUTE, state)
        return state

    @staticmethod
    def _session_will_clear() -> bool:
        # Do not recreate a deleted user_session merely to cache durable state.
        return getattr(chainlit_context.session, "to_clear", False) is True

    async def _publish_response(
        self,
        response: Response,
        *,
        model_id: str,
        ledger_message: cl.Message | None = None,
    ) -> PendingHitl | None:
        raise_for_response(response)
        calls = function_calls(response)
        if calls:
            return await persist_pending_hitl(
                codec=self._codec,
                ledger_message=ledger_message,
                model_id=model_id,
                response_id=response.id,
                function_calls=calls,
                prompt=self._prompt(calls),
                store_in_session=not self._session_will_clear(),
            )
        if ledger_message is not None:
            await complete_pending_hitl(
                ledger_message,
                codec=self._codec,
                store_in_session=not self._session_will_clear(),
            )
        await self._publish_final(response)
        return None

    async def _reopen(self, pending: PendingHitl) -> None:
        if cl.user_session.get(PENDING_HITL_SESSION_KEY) is not pending:
            return
        task_started = False
        try:
            await chainlit_context.emitter.task_start()
            task_started = True
            await self._resolve(pending)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Chainlit HITL automatic resume failed")
            await send_ui_message(f"Response failed: {exc}")
        finally:
            if task_started:
                await chainlit_context.emitter.task_end()


async def resolve_hitl(
    pending: PendingHitl,
    *,
    ask: _AskForOutput,
    continue_response: _ContinueResponse,
    publish_response: _PublishResponse,
) -> None:
    """Resolve complete HITL batches until completion or cancelled input."""
    while True:
        continuation = pending.continuation
        decisions = []
        for call in continuation.function_calls:
            decision = await ask(call, pending.message)
            if decision is None:
                return
            decisions.append(decision)

        response = await continue_response(
            function_call_outputs(continuation, decisions),
            model_id=continuation.model_id,
            previous_response_id=continuation.response_id,
        )
        next_pending = await publish_response(
            response,
            model_id=continuation.model_id,
            ledger_message=pending.message,
        )
        if next_pending is None:
            return
        pending = next_pending


async def persist_pending_hitl(
    *,
    codec: HitlLedgerCodec,
    ledger_message: cl.Message | None,
    model_id: str,
    response_id: str,
    function_calls: Sequence[ResponseFunctionToolCall],
    prompt: str,
    metadata_key: str = HITL_LEDGER_METADATA_KEY,
    session_key: str = PENDING_HITL_SESSION_KEY,
    store_in_session: bool = True,
) -> PendingHitl:
    """Persist the exact continuation before an application solicits input."""
    continuation = codec.continuation(
        model_id=model_id,
        response_id=response_id,
        function_calls=function_calls,
    )
    ledger = codec.pending_metadata(continuation)
    if ledger_message is None:
        ledger_message = cl.Message(content=prompt)
        _set_ledger_metadata(ledger_message, ledger, metadata_key=metadata_key)
        await ledger_message.send()
    else:
        ledger_message.content = prompt
        _set_ledger_metadata(ledger_message, ledger, metadata_key=metadata_key)
        await ledger_message.update()

    pending = PendingHitl(message=ledger_message, continuation=continuation)
    if store_in_session:
        cl.user_session.set(session_key, pending)
    return pending


async def complete_pending_hitl(
    ledger_message: cl.Message,
    *,
    codec: HitlLedgerCodec,
    metadata_key: str = HITL_LEDGER_METADATA_KEY,
    session_key: str = PENDING_HITL_SESSION_KEY,
    store_in_session: bool = True,
) -> None:
    """Persist completion before rendering output so resume cannot replay."""
    _set_ledger_metadata(
        ledger_message,
        codec.completed_metadata(),
        metadata_key=metadata_key,
    )
    await ledger_message.update()
    if store_in_session:
        cl.user_session.set(session_key, None)


def _newest_pending_ledger(
    steps: object,
    *,
    codec: HitlLedgerCodec,
    metadata_key: str = HITL_LEDGER_METADATA_KEY,
) -> _PendingLedgerEntry | None:
    """Select the newest ledger step, with completion blocking older replay."""
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise InvalidHitlLedgerError("Persisted Chainlit steps are invalid.")

    for step in reversed(steps):
        if not isinstance(step, Mapping):
            continue
        metadata = step.get("metadata")
        if not isinstance(metadata, Mapping) or metadata_key not in metadata:
            continue
        continuation = codec.decode(metadata[metadata_key])
        if continuation is None:
            return None
        return _PendingLedgerEntry(step=step, continuation=continuation)
    return None


def restore_pending_hitl(
    thread: ThreadDict,
    *,
    codec: HitlLedgerCodec,
    metadata_key: str = HITL_LEDGER_METADATA_KEY,
) -> PendingHitl | None:
    """Restore the newest pending continuation from a Chainlit thread."""
    entry = _newest_pending_ledger(
        thread.get("steps", []),
        codec=codec,
        metadata_key=metadata_key,
    )
    if entry is None:
        return None

    restored_step = dict(entry.step)
    created_at = restored_step.get("createdAt")
    if isinstance(created_at, str) and not created_at.endswith("Z"):
        # Chainlit's SQL layer can return naive ISO text, while its write path
        # accepts the same timestamp only with an explicit UTC suffix.
        restored_step["createdAt"] = f"{created_at}Z"
    try:
        message = cl.Message.from_dict(cast(StepDict, restored_step))
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidHitlLedgerError(
            "The pending HITL message cannot be restored."
        ) from exc
    return PendingHitl(message=message, continuation=entry.continuation)


async def remove_persisted_custom_elements(
    thread: ThreadDict,
    *,
    step_id: str,
    element_name: str,
) -> None:
    """Remove stale controls before Chainlit rehydrates a resumed thread."""
    elements = thread.get("elements") or []
    stale_elements = [
        element
        for element in elements
        if (
            element.get("id")
            and element.get("type") == "custom"
            and element.get("name") == element_name
            and element.get("forId") == step_id
        )
    ]
    if not stale_elements:
        return

    stale_ids = {element["id"] for element in stale_elements}
    thread["elements"] = [
        element for element in elements if element.get("id") not in stale_ids
    ]
    for element_dict in stale_elements:
        element = cl.CustomElement.from_dict(element_dict)
        await element.remove()


def _set_ledger_metadata(
    message: cl.Message,
    ledger: Mapping[str, object],
    *,
    metadata_key: str,
) -> None:
    # Message.from_dict() shares this mapping with the restored thread step.
    # Preserve its identity so Chainlit's post-hook context rebuild sees updates.
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    mark_model_context_excluded(message)
    metadata.update(message.metadata or {})
    metadata[metadata_key] = dict(ledger)
    message.metadata = metadata


__all__ = [
    "HITL_LEDGER_METADATA_KEY",
    "PENDING_HITL_SESSION_KEY",
    "HitlWorkflow",
    "PendingHitl",
    "complete_pending_hitl",
    "persist_pending_hitl",
    "remove_persisted_custom_elements",
    "resolve_hitl",
    "restore_pending_hitl",
]
