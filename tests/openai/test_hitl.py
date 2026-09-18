import json

import pytest
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_function_tool_call import CallerProgram

from chainlit_utils.openai.hitl import (
    HITL_LEDGER_SCHEMA_VERSION,
    HitlLedgerCodec,
    InvalidHitlLedgerError,
    function_call_outputs,
)

TOOL_NAME = "human_review"


def function_call(
    suffix: str,
    *,
    name: str = TOOL_NAME,
    caller: CallerProgram | None = None,
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{suffix}",
        arguments=json.dumps({"question": f"Approve {suffix}?"}),
        call_id=f"call_{suffix}",
        caller=caller,
        name=name,
        status="completed",
        type="function_call",
    )


def pending_metadata(
    codec: HitlLedgerCodec,
    calls: list[ResponseFunctionToolCall],
) -> dict[str, object]:
    continuation = codec.continuation(
        model_id="review-model",
        response_id="resp_review",
        function_calls=calls,
    )
    return codec.pending_metadata(continuation)


def test_pending_ledger_round_trips_exact_function_call_batch() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    calls = [function_call("one"), function_call("two")]

    restored = codec.decode(pending_metadata(codec, calls))

    assert restored is not None
    assert restored.model_id == "review-model"
    assert restored.response_id == "resp_review"
    assert restored.function_calls == tuple(calls)


def test_completed_ledger_decodes_without_a_continuation() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)

    assert codec.decode(codec.completed_metadata()) is None


@pytest.mark.parametrize(
    "corruption",
    ["future-version", "ledger-extra", "function-call-extra"],
)
def test_ledger_rejects_unknown_or_future_values(corruption: str) -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    raw = pending_metadata(codec, [function_call("one")])
    if corruption == "future-version":
        raw["schema_version"] = HITL_LEDGER_SCHEMA_VERSION + 1
    elif corruption == "ledger-extra":
        raw["future"] = True
    else:
        calls = raw["function_calls"]
        assert isinstance(calls, list)
        call = calls[0]
        assert isinstance(call, dict)
        call["future"] = True

    with pytest.raises(InvalidHitlLedgerError):
        codec.decode(raw)


def test_ledger_rejects_a_different_tool() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)

    with pytest.raises(InvalidHitlLedgerError, match="other_tool"):
        codec.continuation(
            model_id="review-model",
            response_id="resp_review",
            function_calls=[function_call("one", name="other_tool")],
        )


def test_ledger_rejects_duplicate_call_ids() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    raw = pending_metadata(codec, [function_call("one"), function_call("two")])
    calls = raw["function_calls"]
    assert isinstance(calls, list)
    first = calls[0]
    second = calls[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    second["call_id"] = first["call_id"]

    with pytest.raises(InvalidHitlLedgerError, match="call IDs must be unique"):
        codec.decode(raw)


def test_function_outputs_require_the_complete_batch_in_call_order() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    continuation = codec.continuation(
        model_id="review-model",
        response_id="resp_review",
        function_calls=[function_call("one"), function_call("two")],
    )

    assert function_call_outputs(continuation, ["approve", "reject"]) == [
        {
            "type": "function_call_output",
            "call_id": "call_one",
            "output": "approve",
        },
        {
            "type": "function_call_output",
            "call_id": "call_two",
            "output": "reject",
        },
    ]
    with pytest.raises(ValueError, match="Every HITL function call"):
        function_call_outputs(continuation, ["approve"])


def test_function_output_preserves_a_program_caller() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    continuation = codec.continuation(
        model_id="review-model",
        response_id="resp_review",
        function_calls=[
            function_call(
                "one",
                caller=CallerProgram(caller_id="call_program", type="program"),
            )
        ],
    )

    assert function_call_outputs(continuation, ["approve"])[0]["caller"] == {
        "caller_id": "call_program",
        "type": "program",
    }


def test_pending_ledger_uses_openai_wire_aliases() -> None:
    codec = HitlLedgerCodec(TOOL_NAME)
    call = ResponseFunctionToolCall.model_validate(
        {
            "arguments": "{}",
            "async": True,
            "call_id": "call_async",
            "name": TOOL_NAME,
            "type": "function_call",
        }
    )

    raw = pending_metadata(codec, [call])
    persisted_calls = raw["function_calls"]

    assert isinstance(persisted_calls, list)
    assert persisted_calls[0]["async"] is True
    assert "async_" not in persisted_calls[0]
    restored = codec.decode(raw)
    assert restored is not None and restored.function_calls[0].async_ is True
