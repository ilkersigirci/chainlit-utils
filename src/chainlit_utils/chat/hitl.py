"""Persist and restore OpenAI Responses HITL workflows in Chainlit."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import chainlit as cl
from chainlit.step import StepDict
from chainlit.types import ThreadDict
from openai.types.responses import Response, ResponseFunctionToolCall

from chainlit_utils.chat.history import mark_model_context_excluded
from chainlit_utils.openai.hitl import (
    HitlContinuation,
    HitlLedgerCodec,
    InvalidHitlLedgerError,
    function_call_outputs,
)

HITL_LEDGER_METADATA_KEY = "chainlit_utils.hitl_ledger"
PENDING_HITL_SESSION_KEY = "chainlit_utils.pending_hitl"


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
    cl.user_session.set(session_key, pending)
    return pending


async def complete_pending_hitl(
    ledger_message: cl.Message,
    *,
    codec: HitlLedgerCodec,
    metadata_key: str = HITL_LEDGER_METADATA_KEY,
    session_key: str = PENDING_HITL_SESSION_KEY,
) -> None:
    """Persist completion before rendering output so resume cannot replay."""
    _set_ledger_metadata(
        ledger_message,
        codec.completed_metadata(),
        metadata_key=metadata_key,
    )
    await ledger_message.update()
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
    "PendingHitl",
    "complete_pending_hitl",
    "persist_pending_hitl",
    "remove_persisted_custom_elements",
    "resolve_hitl",
    "restore_pending_hitl",
]
