import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.responses import Response, ResponseFunctionToolCall

from chainlit_utils.chat import hitl
from chainlit_utils.openai.hitl import HITL_LEDGER_SCHEMA_VERSION, HitlLedgerCodec
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


def install_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    socket_id: str = "socket-one",
) -> SimpleNamespace:
    session = SimpleNamespace(socket_id=socket_id)
    monkeypatch.setattr(
        hitl,
        "chainlit_context",
        SimpleNamespace(session=session),
    )
    return session


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
    install_context(monkeypatch)
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


async def test_workflow_persists_sequential_batches_before_publishing_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, dict[str, object] | None]] = []
    message = recording_message(writes)
    session = install_chainlit(monkeypatch, message)
    first = Response.model_construct(
        id="resp_one",
        status="completed",
        output=[function_call("one")],
    )
    second = Response.model_construct(
        id="resp_two",
        status="completed",
        output=[function_call("two")],
    )
    final = Response.model_construct(
        id="resp_final",
        status="completed",
        output=[],
    )
    ask = AsyncMock(side_effect=["approve", "reject"])
    continue_response = AsyncMock(side_effect=[second, final])
    publish_final = AsyncMock()
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=ask,
        continue_response=continue_response,
        prompt=lambda calls: f"Review {calls[0].call_id}",
        publish_final=publish_final,
    )

    await workflow.run(first, model_id="review-model")

    assert continue_response.await_args_list[0].args[0] == [
        {
            "type": "function_call_output",
            "call_id": "call_one",
            "output": "approve",
        }
    ]
    assert [
        call.kwargs["previous_response_id"]
        for call in continue_response.await_args_list
    ] == [
        "resp_one",
        "resp_two",
    ]
    assert writes[0][1][hitl.HITL_LEDGER_METADATA_KEY]["response_id"] == "resp_one"
    assert writes[1][1][hitl.HITL_LEDGER_METADATA_KEY]["response_id"] == "resp_two"
    assert writes[2][1][hitl.HITL_LEDGER_METADATA_KEY]["status"] == "completed"
    assert session[hitl.PENDING_HITL_SESSION_KEY] is None
    publish_final.assert_awaited_once_with(final)


async def test_workflow_keeps_the_pending_ledger_when_continuation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, dict[str, object] | None]] = []
    message = recording_message(writes)
    session = install_chainlit(monkeypatch, message)
    pending_response = Response.model_construct(
        id="resp_one",
        status="completed",
        output=[function_call("one")],
    )
    failed_response = Response.model_construct(
        id="resp_failed",
        status="failed",
        output=[],
        error=SimpleNamespace(message="Resume failed"),
    )
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=AsyncMock(return_value="approve"),
        continue_response=AsyncMock(return_value=failed_response),
        prompt=Mock(return_value="Approve one?"),
        publish_final=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="Resume failed"):
        await workflow.run(pending_response, model_id="review-model")

    assert len(writes) == 1
    assert writes[0][1][hitl.HITL_LEDGER_METADATA_KEY]["status"] == "pending"
    assert isinstance(session[hitl.PENDING_HITL_SESSION_KEY], hitl.PendingHitl)


async def test_workflow_keeps_the_pending_ledger_when_prompt_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, dict[str, object] | None]] = []
    message = recording_message(writes)
    session = install_chainlit(monkeypatch, message)
    pending_response = Response.model_construct(
        id="resp_one",
        status="completed",
        output=[function_call("one")],
    )
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=AsyncMock(side_effect=asyncio.CancelledError),
        continue_response=AsyncMock(),
        prompt=Mock(return_value="Approve one?"),
        publish_final=AsyncMock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await workflow.run(pending_response, model_id="review-model")

    assert len(writes) == 1
    assert writes[0][1][hitl.HITL_LEDGER_METADATA_KEY]["status"] == "pending"
    assert isinstance(session[hitl.PENDING_HITL_SESSION_KEY], hitl.PendingHitl)


async def test_workflow_cancel_ignores_an_unrelated_chat_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = install_chainlit(monkeypatch, recording_message([]))
    chat_task = Mock(done=Mock(return_value=False), cancel=Mock())
    monkeypatch.setattr(
        hitl,
        "chainlit_context",
        SimpleNamespace(
            session=SimpleNamespace(
                current_task=chat_task,
                socket_id="socket-one",
            )
        ),
    )
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=AsyncMock(),
        continue_response=AsyncMock(),
        prompt=Mock(),
        publish_final=AsyncMock(),
    )

    workflow.cancel()

    assert hitl.PENDING_HITL_SESSION_KEY not in session
    chat_task.cancel.assert_not_called()


async def test_workflow_cancel_stops_only_the_active_hitl_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_started = asyncio.Event()

    async def wait_for_input(*_args: object) -> str:
        prompt_started.set()
        await asyncio.Future()
        return "unreachable"

    session = install_chainlit(monkeypatch, recording_message([]))
    chainlit_session = SimpleNamespace(socket_id="socket-one", to_clear=False)
    monkeypatch.setattr(
        hitl,
        "chainlit_context",
        SimpleNamespace(session=chainlit_session),
    )
    pending_response = Response.model_construct(
        id="resp_one",
        status="completed",
        output=[function_call("one")],
    )
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=wait_for_input,
        continue_response=AsyncMock(),
        prompt=Mock(return_value="Approve one?"),
        publish_final=AsyncMock(),
    )
    hitl_task = asyncio.create_task(
        workflow.run(pending_response, model_id="review-model")
    )
    await prompt_started.wait()

    workflow.cancel()

    with pytest.raises(asyncio.CancelledError):
        await hitl_task
    assert isinstance(session[hitl.PENDING_HITL_SESSION_KEY], hitl.PendingHitl)


async def test_workflow_defers_an_interrupt_produced_after_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, dict[str, object] | None]] = []
    session = install_chainlit(monkeypatch, recording_message(writes))
    monkeypatch.setattr(
        hitl,
        "chainlit_context",
        SimpleNamespace(session=SimpleNamespace(socket_id="socket-one", to_clear=True)),
    )
    pending_response = Response.model_construct(
        id="resp_one",
        status="completed",
        output=[function_call("one")],
    )
    ask = AsyncMock(return_value="approve")
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=ask,
        continue_response=AsyncMock(),
        prompt=Mock(return_value="Approve one?"),
        publish_final=AsyncMock(),
    )

    workflow.cancel()
    session.clear()  # Chainlit clears user_session when navigating to another thread.
    await workflow.run(pending_response, model_id="review-model")

    assert session == {}
    assert writes[0][1][hitl.HITL_LEDGER_METADATA_KEY]["response_id"] == "resp_one"
    ask.assert_not_awaited()


async def test_workflow_recognizes_a_reconnected_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_chainlit(monkeypatch, recording_message([]))
    chainlit_session = SimpleNamespace(socket_id="socket-one")
    monkeypatch.setattr(
        hitl,
        "chainlit_context",
        SimpleNamespace(session=chainlit_session),
    )
    pending_response = Response.model_construct(
        id="resp_one",
        status="completed",
        output=[function_call("one")],
    )
    ask = AsyncMock(return_value=None)
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=ask,
        continue_response=AsyncMock(),
        prompt=Mock(return_value="Approve one?"),
        publish_final=AsyncMock(),
    )

    workflow.cancel()
    chainlit_session.socket_id = "socket-two"
    await workflow.run(pending_response, model_id="review-model")

    ask.assert_awaited_once()


async def test_workflow_finishes_an_accepted_continuation_after_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuation_started = asyncio.Event()
    release_continuation = asyncio.Event()
    final = Response.model_construct(
        id="resp_final",
        status="completed",
        output=[],
    )

    async def continue_response(*_args: object, **_kwargs: object) -> Response:
        continuation_started.set()
        await release_continuation.wait()
        return final

    session = install_chainlit(monkeypatch, recording_message([]))
    chainlit_session = SimpleNamespace(socket_id="socket-one", to_clear=False)
    monkeypatch.setattr(
        hitl,
        "chainlit_context",
        SimpleNamespace(session=chainlit_session),
    )
    pending_response = Response.model_construct(
        id="resp_one",
        status="completed",
        output=[function_call("one")],
    )
    publish_final = AsyncMock()
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=AsyncMock(return_value="approve"),
        continue_response=continue_response,
        prompt=Mock(return_value="Approve one?"),
        publish_final=publish_final,
    )
    workflow_task = asyncio.create_task(
        workflow.run(pending_response, model_id="review-model")
    )
    await continuation_started.wait()

    chainlit_session.to_clear = True
    workflow.cancel()
    session.clear()
    await asyncio.sleep(0)

    assert not workflow_task.done()
    release_continuation.set()
    await workflow_task
    assert session == {}
    publish_final.assert_awaited_once_with(final)


async def test_workflow_defers_a_later_interrupt_after_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuation_started = asyncio.Event()
    release_continuation = asyncio.Event()
    next_pending_response = Response.model_construct(
        id="resp_two",
        status="completed",
        output=[function_call("two")],
    )

    async def continue_response(*_args: object, **_kwargs: object) -> Response:
        continuation_started.set()
        await release_continuation.wait()
        return next_pending_response

    writes: list[tuple[str, dict[str, object] | None]] = []
    session = install_chainlit(monkeypatch, recording_message(writes))
    chainlit_session = SimpleNamespace(socket_id="socket-one", to_clear=False)
    monkeypatch.setattr(
        hitl,
        "chainlit_context",
        SimpleNamespace(session=chainlit_session),
    )
    first_pending_response = Response.model_construct(
        id="resp_one",
        status="completed",
        output=[function_call("one")],
    )
    ask = AsyncMock(return_value="approve")
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=ask,
        continue_response=continue_response,
        prompt=lambda calls: f"Review {calls[0].call_id}",
        publish_final=AsyncMock(),
    )
    workflow_task = asyncio.create_task(
        workflow.run(first_pending_response, model_id="review-model")
    )
    await continuation_started.wait()

    chainlit_session.to_clear = True
    workflow.cancel()
    session.clear()
    release_continuation.set()
    await workflow_task

    assert session == {}
    assert writes[-1][1][hitl.HITL_LEDGER_METADATA_KEY]["response_id"] == "resp_two"
    assert ask.await_count == 1


async def test_workflow_restores_the_prompt_after_thread_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_context(monkeypatch)
    codec = HitlLedgerCodec(TOOL_NAME)
    ledger = codec.pending_metadata(
        codec.continuation(
            model_id="review-model",
            response_id="resp_one",
            function_calls=[function_call("one")],
        )
    )
    thread = {
        "steps": [
            {
                "id": "ledger-message",
                "createdAt": "2026-09-16T10:00:00",
                "output": "Approve one?",
                "type": "assistant_message",
                "metadata": {hitl.HITL_LEDGER_METADATA_KEY: ledger},
            }
        ],
        "elements": [],
    }
    restored = Mock(id="ledger-message")
    message_factory = SimpleNamespace(from_dict=Mock(return_value=restored))
    session: dict[str, object] = {}
    monkeypatch.setattr(hitl.cl, "Message", message_factory)
    monkeypatch.setattr(
        hitl.cl,
        "user_session",
        SimpleNamespace(
            get=lambda key, default=None: session.get(key, default),
            set=session.__setitem__,
        ),
    )
    schedule = Mock()
    monkeypatch.setattr(hitl, "schedule_after_thread_hydration", schedule)
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=AsyncMock(),
        continue_response=AsyncMock(),
        prompt=Mock(return_value="Approve one?"),
        publish_final=AsyncMock(),
        element_name="Review",
    )

    await workflow.restore(thread)

    pending = session[hitl.PENDING_HITL_SESSION_KEY]
    assert isinstance(pending, hitl.PendingHitl)
    assert pending.message is restored
    assert pending.continuation.response_id == "resp_one"
    schedule.assert_called_once()


async def test_workflow_reports_an_unsupported_persisted_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_context(monkeypatch)
    codec = HitlLedgerCodec(TOOL_NAME)
    ledger = codec.pending_metadata(
        codec.continuation(
            model_id="review-model",
            response_id="resp_one",
            function_calls=[function_call("one")],
        )
    )
    ledger["schema_version"] = HITL_LEDGER_SCHEMA_VERSION + 1
    thread = {"steps": [{"metadata": {hitl.HITL_LEDGER_METADATA_KEY: ledger}}]}
    session: dict[str, object] = {}
    monkeypatch.setattr(
        hitl.cl,
        "user_session",
        SimpleNamespace(
            get=lambda key, default=None: session.get(key, default),
            set=session.__setitem__,
        ),
    )
    schedule = Mock()
    notify = AsyncMock()
    monkeypatch.setattr(hitl, "schedule_after_thread_hydration", schedule)
    monkeypatch.setattr(hitl, "send_ui_message", notify)
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=AsyncMock(),
        continue_response=AsyncMock(),
        prompt=Mock(),
        publish_final=AsyncMock(),
    )

    await workflow.restore(thread)

    schedule.assert_called_once()
    await schedule.call_args.args[0]()
    notify.assert_awaited_once_with(
        "Response failed: HITL ledger schema is unsupported."
    )
    assert session[hitl.PENDING_HITL_SESSION_KEY] is None


async def test_workflow_reopens_pending_review_instead_of_starting_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_context(monkeypatch)
    pending = continuation(HitlLedgerCodec(TOOL_NAME), "one", response_id="resp_one")
    session = {hitl.PENDING_HITL_SESSION_KEY: pending}
    monkeypatch.setattr(
        hitl.cl,
        "user_session",
        SimpleNamespace(
            get=lambda key, default=None: session.get(key, default),
            set=session.__setitem__,
        ),
    )
    notify = AsyncMock()
    monkeypatch.setattr(hitl, "send_ui_message", notify)
    ask = AsyncMock(return_value=None)
    workflow = hitl.HitlWorkflow(
        TOOL_NAME,
        ask=ask,
        continue_response=AsyncMock(),
        prompt=Mock(),
        publish_final=AsyncMock(),
    )
    trigger = SimpleNamespace(metadata={}, update=AsyncMock())

    handled = await workflow.continue_pending(trigger)

    assert handled is True
    assert trigger.metadata[settings.MODEL_CONTEXT_EXCLUDED_KEY] is True
    trigger.update.assert_awaited_once_with()
    notify.assert_awaited_once_with(
        "Resolve the pending interrupt before starting another request."
    )
    ask.assert_awaited_once_with(
        pending.continuation.function_calls[0], pending.message
    )
