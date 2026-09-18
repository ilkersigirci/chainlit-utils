"""OpenAI Responses function-tool calls and continuation input."""

from collections.abc import Sequence
from typing import Any

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
)
from pydantic import TypeAdapter

_RESPONSE_OUTPUT = TypeAdapter(list[ResponseOutputItem])


def function_calls(response: Response) -> list[ResponseFunctionToolCall]:
    """Return client-owned function calls from a completed Response."""
    return [
        item for item in response.output if isinstance(item, ResponseFunctionToolCall)
    ]


def function_call_output(
    call: ResponseFunctionToolCall,
    output: str,
) -> dict[str, Any]:
    """Build a function result while preserving its Responses caller."""
    if not isinstance(output, str):
        raise TypeError("Function call output must be a string.")
    item: dict[str, Any] = {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": output,
    }
    if call.caller is not None:
        item["caller"] = call.caller.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
    return item


def continuation_input(
    response: Response,
    outputs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replay complete output items followed by matching function results."""
    return [
        *_RESPONSE_OUTPUT.dump_python(
            response.output,
            mode="json",
            by_alias=True,
            exclude_none=True,
        ),
        *outputs,
    ]


__all__ = ["continuation_input", "function_call_output", "function_calls"]
