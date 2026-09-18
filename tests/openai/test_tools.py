"""Responses function-tool serialization tests."""

import pytest
from openai.types.responses import (
    Response,
    ResponseCustomToolCall,
    ResponseCustomToolCallOutputItem,
    ResponseFunctionToolCall,
)
from openai.types.responses.parsed_response import ParsedResponseFunctionToolCall
from openai.types.responses.response_function_tool_call import CallerProgram

from chainlit_utils.openai import tools


def _response(*output: object) -> Response:
    return Response.model_construct(status="completed", output=list(output))


def test_function_calls_select_only_client_function_calls() -> None:
    function_call = ResponseFunctionToolCall(
        id="fc-1",
        call_id="call-1",
        name="lookup",
        arguments="{}",
        status="completed",
        type="function_call",
    )
    server_call = ResponseCustomToolCall(
        id="ctc-1",
        call_id="server-call-1",
        name="package_version",
        input="openai",
        status="completed",
        type="custom_tool_call",
    )
    server_output = ResponseCustomToolCallOutputItem(
        id="ctco-1",
        call_id="server-call-1",
        output="openai==installed-version",
        status="completed",
        type="custom_tool_call_output",
    )

    assert tools.function_calls(
        _response(server_call, server_output, function_call)
    ) == [function_call]


@pytest.mark.parametrize("parsed", [False, True])
def test_continuation_replays_wire_fields_without_sdk_only_data(parsed: bool) -> None:
    call = ResponseFunctionToolCall.model_validate(
        {
            "id": "fc-1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query":"status"}',
            "async": True,
            "status": "completed",
            "type": "function_call",
        }
    )
    wire = call.model_dump(mode="json", by_alias=True, exclude_none=True)
    if parsed:
        call = ParsedResponseFunctionToolCall(
            **wire, parsed_arguments={"query": "status"}
        )
    output = {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "done",
    }

    assert tools.continuation_input(_response(call), [output]) == [wire, output]


def test_function_output_preserves_programmatic_caller() -> None:
    call = ResponseFunctionToolCall(
        call_id="call-1",
        name="lookup",
        arguments="{}",
        caller=CallerProgram(caller_id="call-program", type="program"),
        type="function_call",
    )

    assert tools.function_call_output(call, "done") == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "done",
        "caller": {"caller_id": "call-program", "type": "program"},
    }
