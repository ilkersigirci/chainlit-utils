import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import chainlit as cl
import chainlit.socket as socket
import pytest
from chainlit.context import ChainlitContext, init_http_context, init_ws_context
from chainlit.session import WebsocketSession
from chainlit.user_session import user_sessions
from openai.types.responses import Response, ResponseFunctionToolCall
from openai.types.responses.response_function_tool_call import CallerProgram
from pydantic import ValidationError

from chainlit_utils.chat import hitl
from chainlit_utils.settings import settings

TOOL_NAME = "human_review"
ELEMENT_NAME = "HumanReview"
ACTION_NAME = "human_review_submit"


@pytest.fixture
async def chainlit_context():
    context = init_http_context()
    cl.chat_context.clear()
    try:
        yield context
    finally:
        cl.chat_context.clear()
        user_sessions.pop(context.session.id, None)


@pytest.fixture
async def browser_context() -> AsyncIterator[ChainlitContext]:
    """A browser session's context, as Chainlit's action route sets it."""
    session = WebsocketSession(
        id="session",
        socket_id="socket",
        emit=AsyncMock(),
        emit_call=AsyncMock(),
        user_env={},
        client_type="webapp",
        thread_id="thread",
    )
    context = init_ws_context(session)
    cl.chat_context.clear()
    try:
        yield context
    finally:
        cl.chat_context.clear()
        user_sessions.pop(session.id, None)
        await session.delete()


def ui_events(context: ChainlitContext) -> list[str]:
    """Return the events Chainlit sent to the browser, in order."""
    return [call.args[0] for call in context.session.emit.await_args_list]


@pytest.fixture
def removed_controls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the IDs of controls removed from the UI and data layer."""
    removed: list[str] = []

    async def remove(element: cl.CustomElement) -> None:
        removed.append(element.id)

    monkeypatch.setattr(cl.CustomElement, "remove", remove)
    return removed


def function_call(suffix: str, **fields: object) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{suffix}",
        arguments=json.dumps({"question": f"Approve {suffix}?"}),
        call_id=f"call_{suffix}",
        name=TOOL_NAME,
        status="completed",
        type="function_call",
        **fields,
    )


def response(response_id: str, *output: object) -> Response:
    return Response.model_construct(
        id=response_id,
        status="completed",
        output=list(output),
    )


def workflow(
    *,
    continue_response: AsyncMock | None = None,
    publish_final: AsyncMock | None = None,
    validate_outputs=None,
    receipt=None,
) -> tuple[hitl.HitlWorkflow, AsyncMock, AsyncMock]:
    continue_response = continue_response or AsyncMock()
    publish_final = publish_final or AsyncMock()
    instance = hitl.HitlWorkflow(
        TOOL_NAME,
        action_name=ACTION_NAME,
        continue_response=continue_response,
        element_name=ELEMENT_NAME,
        prompt=lambda calls: f"Review {len(calls)} request(s).",
        publish_final=publish_final,
        review=lambda calls: {"reviews": [{"call_id": call.call_id} for call in calls]},
        validate_outputs=validate_outputs,
        receipt=receipt,
    )
    return instance, continue_response, publish_final


def control(pending: hitl.PendingHitl) -> dict[str, str]:
    raw_control = pending.message.elements[0].props[hitl.HITL_CONTROL_PROP]
    assert isinstance(raw_control, dict)
    return raw_control


def stored_ledger(message: cl.Message) -> hitl.HitlLedger:
    return hitl.HitlLedger.model_validate(
        message.metadata[hitl.HITL_LEDGER_METADATA_KEY]
    )


def review_action(
    pending: hitl.PendingHitl, *outputs: str, cancel: bool = False
) -> cl.Action:
    """Build the action the review element submits: its answers, or a cancel."""
    review_control = control(pending)
    payload = {
        key: review_control[key] for key in ("step_id", "element_id", "revision")
    }
    decision = {"cancel": True} if cancel else {"outputs": list(outputs)}
    return cl.Action(name=ACTION_NAME, payload={**payload, **decision})


async def answer(
    instance: hitl.HitlWorkflow, pending: hitl.PendingHitl, *outputs: str
) -> hitl.PendingHitl | None:
    """Submit answers and await their continuation."""
    return await instance.submit(**review_action(pending, *outputs).payload)


async def test_publish_persists_a_review_message_with_its_ledger_and_control(
    chainlit_context,
) -> None:
    instance, continue_response, publish_final = workflow()
    calls = (function_call("one"), function_call("two"))

    pending = await instance.publish(
        response("resp_one", *calls), model_id="review-model"
    )

    assert pending is not None
    assert cl.chat_context.get() == [pending.message]
    assert pending.message.content == "Review 2 request(s)."
    assert pending.message.metadata[settings.MODEL_CONTEXT_EXCLUDED_KEY] is True
    assert stored_ledger(pending.message) == pending.ledger
    assert pending.ledger.function_calls == calls
    assert pending.ledger.outputs is None
    (element,) = pending.message.elements
    assert isinstance(element, cl.CustomElement)
    assert (element.id, element.name) == (pending.ledger.element_id, ELEMENT_NAME)
    assert element.props["reviews"] == [
        {"call_id": "call_one"},
        {"call_id": "call_two"},
    ]
    assert control(pending) == {
        "action": ACTION_NAME,
        "element_id": element.id,
        "revision": "resp_one",
        "step_id": pending.message.id,
    }
    # The browser renders the element from its serialized props.
    assert json.loads(str(element.content)) == element.props
    continue_response.assert_not_awaited()
    publish_final.assert_not_awaited()


async def test_new_user_turn_is_blocked_while_a_review_is_pending(
    chainlit_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _, _ = workflow()
    await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    notify = AsyncMock()
    monkeypatch.setattr(hitl, "send_ui_message", notify)
    trigger = SimpleNamespace(metadata={"host": True}, update=AsyncMock())

    handled = await instance.block_new_message(trigger)

    assert handled is True
    assert trigger.metadata == {
        "host": True,
        settings.MODEL_CONTEXT_EXCLUDED_KEY: True,
    }
    trigger.update.assert_awaited_once_with()
    notify.assert_awaited_once_with(
        "Answer or cancel the pending review before starting another request."
    )


async def test_submit_answers_every_call_and_removes_the_control(
    chainlit_context,
    removed_controls: list[str],
) -> None:
    final = response("resp_final")
    continue_response = AsyncMock(return_value=final)
    instance, _, publish_final = workflow(continue_response=continue_response)
    pending = await instance.publish(
        response("resp_one", function_call("one"), function_call("two")),
        model_id="review-model",
    )
    assert pending is not None

    result = await answer(instance, pending, " approve ", "reject")

    assert result is None
    continue_response.assert_awaited_once_with(
        [
            {
                "type": "function_call_output",
                "call_id": "call_one",
                "output": " approve ",
            },
            {
                "type": "function_call_output",
                "call_id": "call_two",
                "output": "reject",
            },
        ],
        model_id="review-model",
        previous_response_id="resp_one",
    )
    assert stored_ledger(pending.message).outputs == (" approve ", "reject")
    assert pending.message.elements == []
    assert removed_controls == [pending.ledger.element_id]
    publish_final.assert_awaited_once_with(final)


async def test_answered_review_keeps_its_request_and_records_the_decision(
    chainlit_context,
) -> None:
    instance, _, _ = workflow(
        continue_response=AsyncMock(return_value=response("resp_final")),
        receipt=lambda outputs: f"Answered: {', '.join(outputs)}",
    )
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None

    await answer(instance, pending, "approve")

    assert pending.message.content == "Review 1 request(s).\n\nAnswered: approve"


async def test_later_requests_replay_answered_reviews_in_place(
    chainlit_context,
) -> None:
    during_continuation: list[list[dict[str, object]]] = []

    async def continue_response(*_args, **_kwargs) -> Response:
        during_continuation.append(instance.response_input())
        return response("resp_final")

    instance, _, _ = workflow(continue_response=continue_response)
    await cl.Message(content="Refund ORDER-123", type="user_message").send()
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None
    # A pending review has no answer to replay yet.
    assert instance.response_input() == [
        {"role": "user", "content": "Refund ORDER-123"}
    ]

    await answer(instance, pending, "approve")
    await cl.Message(content="Refund approved.").send()

    # Like a stateless Responses transcript, the review replays as its function
    # call and output where it happened, and its UI text does not.
    call = function_call("one").model_dump(mode="json", exclude_none=True)
    output = {
        "type": "function_call_output",
        "call_id": "call_one",
        "output": "approve",
    }
    assert instance.response_input() == [
        {"role": "user", "content": "Refund ORDER-123"},
        call,
        output,
        {"role": "assistant", "content": "Refund approved.", "phase": "final_answer"},
    ]
    # The answers are recorded before the continuation runs, so it replays the
    # review too.
    assert during_continuation == [
        [{"role": "user", "content": "Refund ORDER-123"}, call, output]
    ]


async def test_cancel_closes_the_review_without_continuing(
    chainlit_context,
    removed_controls: list[str],
) -> None:
    instance, continue_response, publish_final = workflow()
    await cl.Message(content="Refund ORDER-123", type="user_message").send()
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None
    action = review_action(pending, cancel=True)

    reply = await instance.submit_action(action)

    # Like a cancelled Open WebUI card, the review records it and the graph does
    # not resume; the thread takes new messages, and nothing of it replays.
    assert reply == {"ok": True}
    assert pending.message.content == "Review 1 request(s).\n\nCancelled."
    assert stored_ledger(pending.message).cancelled is True
    assert removed_controls == [pending.ledger.element_id]
    assert chainlit_context.session.current_task is None
    continue_response.assert_not_awaited()
    publish_final.assert_not_awaited()
    trigger = SimpleNamespace(metadata={}, update=AsyncMock())
    assert await instance.block_new_message(trigger) is False
    assert instance.response_input() == [
        {"role": "user", "content": "Refund ORDER-123"}
    ]
    assert await instance.submit_action(action) == {
        "ok": False,
        "error": "This human-review request is no longer pending.",
    }


async def test_client_function_calls_after_review_go_to_publish_final(
    chainlit_context,
) -> None:
    client_call = function_call("lookup").model_copy(update={"name": "lookup"})
    after_review = response("resp_lookup", client_call)
    instance, _, publish_final = workflow(
        continue_response=AsyncMock(return_value=after_review)
    )
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None

    assert await answer(instance, pending, "approve") is None

    assert stored_ledger(pending.message).outputs == ("approve",)
    publish_final.assert_awaited_once_with(after_review)


async def test_a_later_pause_asks_in_a_new_message(
    chainlit_context,
    removed_controls: list[str],
) -> None:
    final = response("resp_final")
    continue_response = AsyncMock(
        side_effect=[response("resp_two", function_call("two")), final]
    )
    instance, _, publish_final = workflow(continue_response=continue_response)
    first = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert first is not None

    second = await answer(instance, first, "approve")

    assert second is not None
    # Like a chat transcript, the answered review stays above the new one.
    assert cl.chat_context.get() == [first.message, second.message]
    assert stored_ledger(first.message).outputs == ("approve",)
    assert first.message.elements == []
    assert removed_controls == [first.ledger.element_id]
    assert control(second)["revision"] == "resp_two"
    assert second.message.elements[0].props["reviews"] == [{"call_id": "call_two"}]
    publish_final.assert_not_awaited()

    await answer(instance, second, "reject")

    assert continue_response.await_count == 2
    publish_final.assert_awaited_once_with(final)


async def test_submit_action_answers_at_once_and_continues_as_the_session_task(
    browser_context: ChainlitContext,
    removed_controls: list[str],
) -> None:
    release = asyncio.Event()

    async def continue_response(*_args, **_kwargs) -> Response:
        await release.wait()
        return response("resp_final")

    instance, _, publish_final = workflow(continue_response=continue_response)
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None
    action = review_action(pending, "approve")

    reply = await instance.submit_action(action)
    task = browser_context.session.current_task
    await asyncio.sleep(0)

    # The form gets its reply and becomes a record while the graph still runs;
    # like a chat message, the run is the session's task.
    assert reply == {"ok": True}
    assert stored_ledger(pending.message).outputs == ("approve",)
    assert removed_controls == [pending.ledger.element_id]
    assert task is not None and not task.done()
    assert ui_events(browser_context)[-1] == "task_start"
    # A second click is answered at once instead of waiting for the run.
    assert await instance.submit_action(action) == {
        "ok": False,
        "error": "This human-review request is no longer pending.",
    }

    release.set()
    await task

    publish_final.assert_awaited_once_with(response("resp_final"))
    assert ui_events(browser_context)[-1] == "task_end"


async def test_chainlit_stop_cancels_a_running_continuation(
    browser_context: ChainlitContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def continue_response(*_args, **_kwargs) -> Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    instance, _, publish_final = workflow(continue_response=continue_response)
    notice = AsyncMock()
    monkeypatch.setattr(hitl, "send_ui_message", notice)
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None
    await instance.submit_action(review_action(pending, "approve"))
    task = browser_context.session.current_task
    assert task is not None
    await asyncio.sleep(0)

    # The Stop button reaches Chainlit's own handler, which cancels the
    # session's task. This fails if a Chainlit upgrade stops honoring it.
    await socket.sio.handlers["/"]["stop"](browser_context.session.socket_id)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert ui_events(browser_context)[-1] == "task_end"
    notice.assert_not_awaited()
    publish_final.assert_not_awaited()


async def test_a_failed_continuation_leaves_the_review_answered(
    chainlit_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = Response.model_construct(
        id="resp_failed",
        status="failed",
        output=[],
        error=SimpleNamespace(message="Resume failed"),
    )
    instance, _, publish_final = workflow(
        continue_response=AsyncMock(return_value=failed)
    )
    notice = AsyncMock()
    monkeypatch.setattr(hitl, "send_ui_message", notice)
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None

    assert await instance.submit_action(review_action(pending, "approve")) == {
        "ok": True
    }
    await chainlit_context.session.current_task

    # The chat reports the failure; the next message continues the conversation.
    assert stored_ledger(pending.message).outputs == ("approve",)
    notice.assert_awaited_once_with("Response failed: Resume failed")
    publish_final.assert_not_awaited()


@pytest.mark.parametrize("field", ["step_id", "element_id", "revision"])
async def test_stale_or_forged_submission_is_rejected_before_continuation(
    chainlit_context,
    field: str,
) -> None:
    instance, continue_response, _ = workflow()
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None
    submission = {**review_action(pending, "approve").payload, field: "forged"}

    with pytest.raises(hitl.InvalidHitlSubmissionError, match="stale"):
        await instance.submit(**submission)

    continue_response.assert_not_awaited()


async def test_application_validation_runs_against_trusted_calls(
    chainlit_context,
) -> None:
    validate = Mock(side_effect=ValueError("That decision is not allowed."))
    instance, continue_response, _ = workflow(validate_outputs=validate)
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None

    with pytest.raises(hitl.InvalidHitlSubmissionError, match="not allowed"):
        await answer(instance, pending, "forged")

    validate.assert_called_once_with(pending.ledger.function_calls, ["forged"])
    continue_response.assert_not_awaited()


async def test_resumed_thread_continues_from_its_persisted_ledger(
    chainlit_context,
    removed_controls: list[str],
) -> None:
    final = response("resp_final")
    instance, continue_response, publish_final = workflow(
        continue_response=AsyncMock(
            side_effect=[response("resp_two", function_call("two")), final]
        )
    )
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None
    step_dict = pending.message.to_dict()
    step_dict["createdAt"] = "2026-09-18T13:48:54"

    # A reload restores the message from the data layer without its elements.
    cl.chat_context.clear()
    restored = cl.Message.from_dict(step_dict)
    cl.chat_context.add(restored)
    assert restored.elements == []

    second = await answer(instance, pending, "approve")

    assert second is not None
    assert cl.chat_context.get() == [restored, second.message]
    assert stored_ledger(restored).outputs == ("approve",)
    assert removed_controls == [pending.ledger.element_id]
    assert restored.created_at == "2026-09-18T13:48:54.000000Z"
    publish_final.assert_not_awaited()

    await answer(instance, second, "reject")

    assert continue_response.await_count == 2
    publish_final.assert_awaited_once_with(final)


async def test_terminal_response_is_published_without_a_review(
    chainlit_context,
) -> None:
    final = response("resp_final")
    instance, continue_response, publish_final = workflow()

    result = await instance.publish(final, model_id="review-model")

    assert result is None
    assert cl.chat_context.get() == []
    continue_response.assert_not_awaited()
    publish_final.assert_awaited_once_with(final)


async def test_a_review_cannot_include_other_function_calls(
    chainlit_context,
) -> None:
    client_call = function_call("lookup").model_copy(update={"name": "lookup"})
    instance, _, publish_final = workflow()

    with pytest.raises(ValueError, match="other function calls"):
        await instance.publish(
            response("resp_one", function_call("one"), client_call),
            model_id="review-model",
        )

    assert cl.chat_context.get() == []
    publish_final.assert_not_awaited()


def test_stored_ledger_round_trips_the_exact_calls() -> None:
    calls = (
        function_call(
            "one", caller=CallerProgram(caller_id="call_program", type="program")
        ),
        function_call("two"),
    )
    ledger = hitl.HitlLedger(
        model_id="review-model",
        response_id="resp_one",
        element_id="element",
        function_calls=calls,
        outputs=("approve", "reject"),
    )

    stored = json.loads(
        json.dumps(ledger.model_dump(mode="json", by_alias=True, exclude_none=True))
    )

    assert hitl.HitlLedger.model_validate(stored) == ledger
    assert ledger.output_items(("approve", "reject"))[0]["caller"] == {
        "caller_id": "call_program",
        "type": "program",
    }


@pytest.mark.parametrize(
    "corruption",
    [
        {"future": True},
        {"schema_version": 4},
        {"outputs": ["approve"]},
        {
            "function_calls": [
                function_call("one").model_dump(),
                function_call("one").model_dump(),
            ]
        },
        {"cancelled": True, "outputs": ["approve", "reject"]},
    ],
    ids=[
        "unknown-field",
        "other-version",
        "missing-output",
        "duplicate-call-id",
        "cancelled-with-outputs",
    ],
)
def test_stored_ledger_rejects_corrupt_state(corruption: dict[str, object]) -> None:
    ledger = {
        "model_id": "review-model",
        "response_id": "resp_one",
        "element_id": "element",
        "function_calls": [
            function_call("one").model_dump(),
            function_call("two").model_dump(),
        ],
    }

    with pytest.raises(ValidationError):
        hitl.HitlLedger.model_validate({**ledger, **corruption})


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"model_id": "browser-controlled"}, "Invalid human-review submission."),
        ({"outputs": "approve"}, "Invalid human-review submission."),
        ({"revision": "resp_old"}, "This human-review request is stale."),
        ({"cancel": True}, "Invalid human-review submission."),
        ({"outputs": None}, "Invalid human-review submission."),
    ],
    ids=["extra-field", "text-outputs", "stale", "answer-and-cancel", "neither"],
)
async def test_submit_action_replies_with_the_error_the_element_shows(
    chainlit_context,
    change: dict[str, object],
    error: str,
) -> None:
    instance, continue_response, _ = workflow()
    pending = await instance.publish(
        response("resp_one", function_call("one")), model_id="review-model"
    )
    assert pending is not None
    payload = {**review_action(pending, "approve").payload, **change}

    result = await instance.submit_action(cl.Action(name=ACTION_NAME, payload=payload))

    assert result == {"ok": False, "error": error}
    assert stored_ledger(pending.message).outputs is None
    assert chainlit_context.session.current_task is None
    continue_response.assert_not_awaited()
