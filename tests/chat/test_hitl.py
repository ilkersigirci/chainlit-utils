import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.responses import Response, ResponseFunctionToolCall

from chainlit_utils.chat import hitl
from chainlit_utils.openai.hitl import HitlLedgerCodec
from chainlit_utils.settings import settings

TOOL_NAME = "human_review"


def function_call(suffix: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{suffix}",
        arguments=json.dumps({"question": f"Approve {suffix}?"}),
        call_id=f"call_{suffix}",
        name=TOOL_NAME,
        status="completed",
        type="function_call",
    )


def continuation(
    codec: HitlLedgerCodec,
    suffix: str,
    *,
    response_id: str,
) -> hitl.PendingHitl:
    return hitl.PendingHitl(
        message=Mock(id="ledger-message"),
        continuation=codec.continuation(
            model_id="review-model",
            response_id=response_id,
            function_calls=[function_call(suffix)],
        ),
    )


def recording_message(
    writes: list[tuple[str, dict[str, object] | None]],
    *,
    content: str = "",
    metadata: dict[str, object] | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, metadata=metadata, id="ledger-message")

    async def send() -> SimpleNamespace:
        writes.append(("send", deepcopy(message.metadata)))
        return message

    async def update() -> bool:
        writes.append(("update", deepcopy(message.metadata)))
        return True

    message.send = AsyncMock(side_effect=send)
    message.update = AsyncMock(side_effect=update)
    return message


def install_chainlit(
    monkeypatch: pytest.MonkeyPatch,
    message: SimpleNamespace,
) -> dict[str, object]:
    message_factory = Mock(return_value=message)
    monkeypatch.setattr(hitl.cl, "Message", message_factory)
    session: dict[str, object] = {}
    monkeypatch.setattr(
        hitl.cl,
        "user_session",
        SimpleNamespace(
            get=lambda key, default=None: session.get(key, default),
            set=session.__setitem__,
        ),
    )
    return session


async def test_pending_and_completed_ledgers_are_persisted_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    writes: list[tuple[str, dict[str, object] | None]] = []
    original_metadata = {"host": True}
    message = recording_message(writes, metadata=original_metadata)
    session = install_chainlit(monkeypatch, message)

    pending = await hitl.persist_pending_hitl(
        codec=codec,
        ledger_message=None,
        model_id="review-model",
        response_id="resp_one",
        function_calls=[function_call("one")],
        prompt="Approve one?",
    )

    assert pending.message is message
    assert message.metadata is original_metadata
    assert message.metadata["host"] is True
    assert message.metadata[settings.MODEL_CONTEXT_EXCLUDED_KEY] is True
    assert writes[0][0] == "send"
    assert writes[0][1][hitl.HITL_LEDGER_METADATA_KEY]["response_id"] == "resp_one"
    assert session[hitl.PENDING_HITL_SESSION_KEY] is pending

    next_pending = await hitl.persist_pending_hitl(
        codec=codec,
        ledger_message=message,
        model_id="review-model",
        response_id="resp_two",
        function_calls=[function_call("two")],
        prompt="Approve two?",
    )
    assert writes[1][0] == "update"
    assert writes[1][1][hitl.HITL_LEDGER_METADATA_KEY]["response_id"] == "resp_two"
    assert session[hitl.PENDING_HITL_SESSION_KEY] is next_pending

    await hitl.complete_pending_hitl(message, codec=codec)
    assert writes[2][1][hitl.HITL_LEDGER_METADATA_KEY]["status"] == "completed"
    assert session[hitl.PENDING_HITL_SESSION_KEY] is None


def test_restore_completion_blocks_older_pending_replay() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    pending = codec.pending_metadata(
        codec.continuation(
            model_id="review-model",
            response_id="resp_one",
            function_calls=[function_call("one")],
        )
    )
    steps = [
        {"metadata": {hitl.HITL_LEDGER_METADATA_KEY: pending}},
        {"newer-unrelated-step": True},
        {
            "metadata": {
                hitl.HITL_LEDGER_METADATA_KEY: codec.completed_metadata(),
            }
        },
    ]

    assert hitl.restore_pending_hitl({"steps": steps}, codec=codec) is None


def test_restore_pending_hitl_reconstructs_the_host_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    ledger = codec.pending_metadata(
        codec.continuation(
            model_id="review-model",
            response_id="resp_one",
            function_calls=[function_call("one")],
        )
    )
    step = {
        "id": "ledger-message",
        "createdAt": "2026-09-16T10:00:00",
        "output": "Approve one?",
        "type": "assistant_message",
        "metadata": {hitl.HITL_LEDGER_METADATA_KEY: ledger, "host": True},
        "host-added": {"future": "value"},
    }
    restored_message = Mock(id="ledger-message")
    from_dict = Mock(return_value=restored_message)
    monkeypatch.setattr(hitl.cl, "Message", SimpleNamespace(from_dict=from_dict))

    pending = hitl.restore_pending_hitl({"steps": [step]}, codec=codec)

    assert pending is not None
    assert pending.message is restored_message
    assert pending.continuation.response_id == "resp_one"
    restored_step = from_dict.call_args.args[0]
    assert restored_step["createdAt"] == "2026-09-16T10:00:00Z"


async def test_resolve_hitl_continues_sequential_batches() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    first = continuation(codec, "one", response_id="resp_one")
    second = continuation(codec, "two", response_id="resp_two")
    second = hitl.PendingHitl(message=first.message, continuation=second.continuation)
    continued_responses = [
        Response.model_construct(id="resp_two", status="completed", output=[]),
        Response.model_construct(id="resp_final", status="completed", output=[]),
    ]
    ask = AsyncMock(side_effect=["approve", "reject"])
    continue_response = AsyncMock(side_effect=continued_responses)
    publish_response = AsyncMock(side_effect=[second, None])

    await hitl.resolve_hitl(
        first,
        ask=ask,
        continue_response=continue_response,
        publish_response=publish_response,
    )

    assert continue_response.await_args_list[0].args[0] == [
        {
            "type": "function_call_output",
            "call_id": "call_one",
            "output": "approve",
        }
    ]
    assert continue_response.await_args_list[0].kwargs == {
        "model_id": "review-model",
        "previous_response_id": "resp_one",
    }
    assert continue_response.await_args_list[1].kwargs["previous_response_id"] == (
        "resp_two"
    )
    assert publish_response.await_count == 2


async def test_cancelled_batch_is_not_partially_submitted() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    pending = hitl.PendingHitl(
        message=Mock(id="ledger-message"),
        continuation=codec.continuation(
            model_id="review-model",
            response_id="resp_one",
            function_calls=[function_call("one"), function_call("two")],
        ),
    )
    ask = AsyncMock(side_effect=["approve", None])
    continue_response = AsyncMock()
    publish_response = AsyncMock()

    await hitl.resolve_hitl(
        pending,
        ask=ask,
        continue_response=continue_response,
        publish_response=publish_response,
    )

    continue_response.assert_not_awaited()
    publish_response.assert_not_awaited()


async def test_remove_persisted_custom_elements_removes_only_stale_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = {
        "id": "element-stale",
        "type": "custom",
        "name": "Review",
        "forId": "ledger-message",
    }
    other = {
        "id": "element-other",
        "type": "custom",
        "name": "Review",
        "forId": "another-message",
    }
    thread = {"steps": [], "elements": [stale, other]}
    element = SimpleNamespace(remove=AsyncMock())
    from_dict = Mock(return_value=element)
    monkeypatch.setattr(
        hitl.cl,
        "CustomElement",
        SimpleNamespace(from_dict=from_dict),
    )

    await hitl.remove_persisted_custom_elements(
        thread,
        step_id="ledger-message",
        element_name="Review",
    )

    assert thread["elements"] == [other]
    from_dict.assert_called_once_with(stale)
    element.remove.assert_awaited_once_with()
