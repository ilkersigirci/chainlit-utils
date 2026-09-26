import json

import pytest
from openai.types.responses import ResponseFunctionToolCall

from chainlit_utils.chat.human_review import (
    HUMAN_REVIEW_ELEMENT_NAME,
    HumanReview,
    HumanReviewForm,
)
from chainlit_utils.public_files import PUBLIC_DIR


def _call(suffix: str, **payload: object) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{suffix}",
        call_id=f"call_{suffix}",
        name="human_review",
        arguments=json.dumps(payload),
        status="completed",
        type="function_call",
    )


def _review(call: ResponseFunctionToolCall) -> HumanReview:
    payload = json.loads(call.arguments)
    return HumanReview(
        prompt=payload["question"],
        choices=tuple(payload.get("choices", ())),
        allow_other=payload.get("allow_other", False),
    )


form = HumanReviewForm(_review)
refund = _call("refund", question="Approve refund?", choices=["approve", "reject"])
carrier = _call("carrier", question="Choose carrier")


def test_form_presents_the_complete_batch_in_the_bundled_element() -> None:
    assert form.props([refund, carrier]) == {
        "reviews": [
            {
                "prompt": "Approve refund?",
                "choices": ["approve", "reject"],
                "allow_other": False,
            },
            {"prompt": "Choose carrier", "choices": [], "allow_other": True},
        ]
    }
    assert form.prompt([refund]) == "Approve refund?"
    assert form.prompt([refund, carrier]) == (
        "Human review is required for 2 requests.\n\n"
        "1. Approve refund?\n\n2. Choose carrier"
    )
    assert (PUBLIC_DIR / "elements" / f"{HUMAN_REVIEW_ELEMENT_NAME}.jsx").is_file()


@pytest.mark.parametrize(
    ("calls", "outputs", "result"),
    [
        ([refund, carrier], [" approve ", " by sea "], ("approve", "by sea")),
        (
            [_call("other", question="Approve?", choices=["yes"], allow_other=True)],
            ["only on Monday"],
            ("only on Monday",),
        ),
        ([refund, carrier], ["approve"], "Every human-review request"),
        ([refund, carrier], ["approve", "  "], "non-empty strings"),
        ([refund, carrier], ["forged", "by sea"], "not an allowed choice"),
    ],
    ids=["choice-and-text", "other-answer", "missing", "blank", "forged"],
)
def test_answers_are_checked_against_the_trusted_calls(
    calls: list[ResponseFunctionToolCall],
    outputs: list[str],
    result: tuple[str, ...] | str,
) -> None:
    if isinstance(result, tuple):
        assert form.validate_outputs(calls, outputs) == result
        return
    with pytest.raises(ValueError, match=result):
        form.validate_outputs(calls, outputs)
