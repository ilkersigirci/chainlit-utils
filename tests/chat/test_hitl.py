import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import chainlit as cl
import pytest
from chainlit.context import init_http_context
from chainlit.user_session import user_sessions
from openai.types.responses import Response, ResponseFunctionToolCall

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


def function_call(suffix: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{suffix}",
        arguments=json.dumps({"question": f"Approve {suffix}?"}),
        call_id=f"call_{suffix}",
        name=TOOL_NAME,
        status="completed",
        type="function_call",
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
    )
    return instance, continue_response, publish_final


def control(pending: hitl.PendingHitl) -> dict[str, str]:
    raw_control = pending.message.elements[0].props[hitl.HITL_CONTROL_PROP]
    assert isinstance(raw_control, dict)
    return raw_control


async def test_publish_persists_a_normal_message_and_complete_batch(
    chainlit_context,
) -> None:
    instance, continue_response, publish_final = workflow()

    pending = await instance.publish(
        response("resp_one", function_call("one"), function_call("two")),
        model_id="review-model",
    )

    assert pending is not None
    assert pending.continuation.response_id == "resp_one"
    assert len(pending.continuation.function_calls) == 2
    assert cl.chat_context.get() == [pending.message]
    assert pending.message.metadata[settings.MODEL_CONTEXT_EXCLUDED_KEY] is True
    ledger = pending.message.metadata[hitl.HITL_LEDGER_METADATA_KEY]
    assert ledger["status"] == "pending"
    assert ledger["response_id"] == "resp_one"
    assert len(pending.message.elements) == 1
    element = pending.message.elements[0]
    assert isinstance(element, cl.CustomElement)
    assert element.name == ELEMENT_NAME
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
    assert (
        pending.message.metadata[hitl.HITL_ELEMENT_METADATA_KEY]
        == element.id
        == pending.element_id
    )
    assert json.loads(str(element.content)) == element.props
    continue_response.assert_not_awaited()
    publish_final.assert_not_awaited()


async def test_new_user_turn_is_blocked_by_the_persisted_ledger(
    chainlit_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, _, _ = workflow()
    await instance.publish(
        response("resp_one", function_call("one")),
        model_id="review-model",
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
        "Resolve the pending interrupt before starting another request."
    )


async def test_submit_completes_one_continuation_and_removes_the_control(
    chainlit_context,
) -> None:
    final = response("resp_final")
    continue_response = AsyncMock(return_value=final)
    instance, _, publish_final = workflow(continue_response=continue_response)
    pending = await instance.publish(
        response("resp_one", function_call("one"), function_call("two")),
        model_id="review-model",
    )
    assert pending is not None
    review_control = control(pending)
    element = pending.message.elements[0]
    element.remove = AsyncMock()  # type: ignore[method-assign]

    result = await instance.submit(
        step_id=review_control["step_id"],
        element_id=review_control["element_id"],
        revision=review_control["revision"],
        outputs=[" approve ", "reject"],
    )

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
    ledger = pending.message.metadata[hitl.HITL_LEDGER_METADATA_KEY]
    assert ledger["status"] == "completed"
    assert hitl.HITL_ELEMENT_METADATA_KEY not in pending.message.metadata
    assert pending.message.elements == []
    element.remove.assert_awaited_once_with()
    publish_final.assert_awaited_once_with(final)


async def test_chained_interrupt_updates_the_same_element_one_action_at_a_time(
    chainlit_context,
) -> None:
    second = response("resp_two", function_call("two"))
    final = response("resp_final")
    continue_response = AsyncMock(side_effect=[second, final])
    instance, _, publish_final = workflow(continue_response=continue_response)
    first = await instance.publish(
        response("resp_one", function_call("one")),
        model_id="review-model",
    )
    assert first is not None
    first_control = control(first)
    element_id = first_control["element_id"]

    next_pending = await instance.submit(
        step_id=first_control["step_id"],
        element_id=element_id,
        revision=first_control["revision"],
        outputs=["approve"],
    )

    assert next_pending is not None
    assert continue_response.await_count == 1
    assert next_pending.message is first.message
    assert next_pending.message.elements[0].id == element_id
    next_control = control(next_pending)
    assert next_control["revision"] == "resp_two"
    assert next_pending.message.elements[0].props["reviews"] == [
        {"call_id": "call_two"}
    ]
    publish_final.assert_not_awaited()

    next_pending.message.elements[0].remove = AsyncMock()  # type: ignore[method-assign]
    await instance.submit(
        step_id=next_control["step_id"],
        element_id=next_control["element_id"],
        revision=next_control["revision"],
        outputs=["reject"],
    )

    assert continue_response.await_count == 2
    publish_final.assert_awaited_once_with(final)


async def test_failed_continuation_keeps_the_prior_revision_pending(
    chainlit_context,
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
    pending = await instance.publish(
        response("resp_one", function_call("one")),
        model_id="review-model",
    )
    assert pending is not None
    review_control = control(pending)

    with pytest.raises(RuntimeError, match="Resume failed"):
        await instance.submit(
            step_id=review_control["step_id"],
            element_id=review_control["element_id"],
            revision=review_control["revision"],
            outputs=["approve"],
        )

    ledger = pending.message.metadata[hitl.HITL_LEDGER_METADATA_KEY]
    assert ledger["status"] == "pending"
    assert ledger["response_id"] == "resp_one"
    assert control(pending)["revision"] == "resp_one"
    publish_final.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step_id", "another-step", "stale"),
        ("element_id", "another-element", "control"),
        ("revision", "another-response", "stale"),
    ],
)
async def test_stale_or_forged_submission_is_rejected_before_continuation(
    chainlit_context,
    field: str,
    value: str,
    message: str,
) -> None:
    instance, continue_response, _ = workflow()
    pending = await instance.publish(
        response("resp_one", function_call("one")),
        model_id="review-model",
    )
    assert pending is not None
    submission = {
        "step_id": control(pending)["step_id"],
        "element_id": control(pending)["element_id"],
        "revision": control(pending)["revision"],
        "outputs": ["approve"],
    }
    submission[field] = value

    with pytest.raises(hitl.InvalidHitlSubmissionError, match=message):
        await instance.submit(**submission)

    continue_response.assert_not_awaited()


async def test_application_validation_runs_against_trusted_calls(
    chainlit_context,
) -> None:
    validate = Mock(side_effect=ValueError("That decision is not allowed."))
    instance, continue_response, _ = workflow(validate_outputs=validate)
    pending = await instance.publish(
        response("resp_one", function_call("one")),
        model_id="review-model",
    )
    assert pending is not None
    review_control = control(pending)

    with pytest.raises(hitl.InvalidHitlSubmissionError, match="not allowed"):
        await instance.submit(
            step_id=review_control["step_id"],
            element_id=review_control["element_id"],
            revision=review_control["revision"],
            outputs=["forged"],
        )

    validate.assert_called_once_with(pending.continuation.function_calls, ["forged"])
    continue_response.assert_not_awaited()


async def test_concurrent_same_session_duplicates_continue_only_once(
    chainlit_context,
) -> None:
    continuation_started = asyncio.Event()
    release_continuation = asyncio.Event()

    async def continue_response(*_args, **_kwargs) -> Response:
        continuation_started.set()
        await release_continuation.wait()
        return response("resp_final")

    continue_mock = AsyncMock(side_effect=continue_response)
    instance, _, _ = workflow(continue_response=continue_mock)
    pending = await instance.publish(
        response("resp_one", function_call("one")),
        model_id="review-model",
    )
    assert pending is not None
    review_control = control(pending)
    pending.message.elements[0].remove = AsyncMock()  # type: ignore[method-assign]
    submission = {
        "step_id": review_control["step_id"],
        "element_id": review_control["element_id"],
        "revision": review_control["revision"],
        "outputs": ["approve"],
    }
    start = asyncio.Event()

    async def submit() -> hitl.PendingHitl | None:
        await start.wait()
        return await instance.submit(**submission)

    first = asyncio.create_task(submit())
    duplicate = asyncio.create_task(submit())
    start.set()
    await continuation_started.wait()
    release_continuation.set()
    results = await asyncio.gather(first, duplicate, return_exceptions=True)

    assert continue_mock.await_count == 1
    assert sum(result is None for result in results) == 1
    errors = [result for result in results if isinstance(result, Exception)]
    assert len(errors) == 1
    assert isinstance(errors[0], hitl.InvalidHitlSubmissionError)
    assert "no longer pending" in str(errors[0])


async def test_resumed_thread_reconstructs_its_control_from_persisted_metadata(
    chainlit_context,
) -> None:
    second = response("resp_two", function_call("two"))
    final = response("resp_final")
    instance, continue_response, publish_final = workflow(
        continue_response=AsyncMock(side_effect=[second, final])
    )
    pending = await instance.publish(
        response("resp_one", function_call("one")),
        model_id="review-model",
    )
    assert pending is not None
    review_control = control(pending)
    step_dict = pending.message.to_dict()
    assert step_dict["createdAt"] is not None
    step_dict["createdAt"] = "2026-09-18T13:48:54"

    cl.chat_context.clear()
    restored_message = cl.Message.from_dict(step_dict)
    cl.chat_context.add(restored_message)
    assert restored_message.elements == []

    next_pending = await instance.submit(
        step_id=review_control["step_id"],
        element_id=review_control["element_id"],
        revision=review_control["revision"],
        outputs=["approve"],
    )

    assert next_pending is not None
    assert next_pending.message is restored_message
    assert next_pending.element_id == review_control["element_id"]
    assert next_pending.message.elements[0].id == review_control["element_id"]
    assert control(next_pending)["revision"] == "resp_two"
    assert next_pending.message.created_at == "2026-09-18T13:48:54.000000Z"
    publish_final.assert_not_awaited()

    next_control = control(next_pending)
    next_pending.message.elements[0].remove = AsyncMock()  # type: ignore[method-assign]
    await instance.submit(
        step_id=next_control["step_id"],
        element_id=next_control["element_id"],
        revision=next_control["revision"],
        outputs=["reject"],
    )

    assert continue_response.await_count == 2
    publish_final.assert_awaited_once_with(final)


async def test_terminal_response_is_published_without_a_ledger(
    chainlit_context,
) -> None:
    final = response("resp_final")
    instance, continue_response, publish_final = workflow()

    result = await instance.publish(final, model_id="review-model")

    assert result is None
    assert cl.chat_context.get() == []
    continue_response.assert_not_awaited()
    publish_final.assert_awaited_once_with(final)
