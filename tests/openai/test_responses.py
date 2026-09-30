from unittest.mock import AsyncMock, Mock

import pytest
from chainlit.config import config
from chainlit.context import init_http_context
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_output_text import AnnotationURLCitation

from chainlit_utils.openai import responses


def _response(*output: object) -> Response:
    return Response.model_construct(status="completed", output=list(output))


def test_response_projection_keeps_supported_text_roles_and_phase() -> None:
    assert responses.response_input(
        [
            {"role": "system", "content": "Rules"},
            {"role": "assistant", "content": "Working", "phase": "commentary"},
            {"role": "assistant", "content": "Answer"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": ["ignored"]},
        ]
    ) == [
        {"role": "system", "content": "Rules"},
        {"role": "assistant", "content": "Working", "phase": "commentary"},
        {"role": "assistant", "content": "Answer", "phase": "final_answer"},
    ]


def test_final_answer_excludes_commentary_and_includes_refusals() -> None:
    output = ResponseOutputMessage(
        id="answer",
        type="message",
        role="assistant",
        status="completed",
        phase="final_answer",
        content=[
            ResponseOutputText(type="output_text", text="Answer. ", annotations=[]),
            ResponseOutputRefusal(type="refusal", refusal="Cannot add more."),
        ],
    )
    commentary = output.model_copy(
        update={
            "id": "commentary",
            "phase": "commentary",
            "content": [
                ResponseOutputText(type="output_text", text="Searching", annotations=[])
            ],
        }
    )

    assert responses.final_answer(_response(commentary, output)) == (
        "Answer. Cannot add more."
    )


def test_citations_become_chainlit_side_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Answer [source]"
    output = ResponseOutputMessage(
        id="answer",
        type="message",
        role="assistant",
        status="completed",
        phase="final_answer",
        content=[
            ResponseOutputText(
                type="output_text",
                text=text,
                annotations=[
                    AnnotationURLCitation(
                        type="url_citation",
                        url="https://example.com/source",
                        title="Source",
                        start_index=text.index("[source]"),
                        end_index=len(text) - 1,
                    )
                ],
            )
        ],
    )
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(responses.cl, "Text", factory)

    assert len(responses.citation_elements(_response(output))) == 1
    factory.assert_called_once_with(
        name="[source]",
        content="[Open source](<https://example.com/source>)",
        display="side",
    )


def test_incomplete_response_reports_native_reason() -> None:
    response = _response().model_copy(
        update={
            "status": "incomplete",
            "incomplete_details": IncompleteDetails(reason="max_output_tokens"),
        }
    )
    with pytest.raises(RuntimeError, match="Response incomplete: max_output_tokens"):
        responses.raise_for_response(response)


async def test_commentary_task_list_completes_previous_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_list = Mock(status="Ready", add_task=AsyncMock(), send=AsyncMock())
    tasks = [Mock(), Mock()]
    monkeypatch.setattr(responses.cl, "TaskList", Mock(return_value=task_list))
    monkeypatch.setattr(responses.cl, "Task", Mock(side_effect=tasks))
    renderer = responses.CommentaryTaskList()

    await renderer.add("Searching")
    await renderer.add("Composing")
    await renderer.complete()

    assert [task.status for task in tasks] == [
        responses.cl.TaskStatus.DONE,
        responses.cl.TaskStatus.DONE,
    ]
    assert task_list.status == "Done"


@pytest.mark.parametrize("stopped", [False, True], ids=["completed", "stopped"])
async def test_commentary_steps_show_progress_and_finish_without_chat_messages(
    monkeypatch: pytest.MonkeyPatch,
    stopped: bool,
) -> None:
    context = init_http_context()
    monkeypatch.setattr(config.ui, "cot", "tool_call")
    sent = AsyncMock()
    updated = AsyncMock()
    monkeypatch.setattr(context.emitter, "send_step", sent)
    monkeypatch.setattr(context.emitter, "update_step", updated)
    renderer = responses.CommentarySteps()

    await renderer.complete()
    await renderer.stop()
    await renderer.add("")
    sent.assert_not_called()
    updated.assert_not_called()

    await renderer.add("Searching")
    await renderer.add("Composing")
    if stopped:
        await renderer.stop()
    else:
        await renderer.complete()
    await renderer.complete()
    await renderer.stop()

    sent.assert_awaited_once()
    first = sent.call_args.args[0]
    progress, last = [call.args[0] for call in updated.call_args_list]
    assert first["name"] == "Searching"
    assert first["output"] == "- Searching"
    assert first["type"] == "tool"
    assert first["start"] and first["end"] is None
    assert first["defaultOpen"] is False
    assert progress["id"] == first["id"] == last["id"]
    assert progress["name"] == last["name"] == "Composing"
    assert progress["output"] == "- Searching\n- Composing"
    assert progress["end"] is None
    assert last["end"] >= last["start"]
    assert last["isError"] is stopped
    assert last["output"] == "- Searching\n- Composing" + (
        "\n- Stopped" if stopped else ""
    )
    assert responses.cl.chat_context.get() == []
