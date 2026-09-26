"""The bundled human-review form for `HitlWorkflow`."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from openai.types.responses import ResponseFunctionToolCall

# Matches public/elements/HumanReview.jsx.
HUMAN_REVIEW_ELEMENT_NAME = "HumanReview"


@dataclass(frozen=True, slots=True)
class HumanReview:
    """One function call's request, as the `HumanReview` element shows it.

    The reviewer picks one of ``choices`` or, when ``allow_other`` is set or no
    choices exist, writes a free-text answer.
    """

    prompt: str
    choices: tuple[str, ...] = ()
    allow_other: bool = False

    def accepts(self, answer: str) -> bool:
        return not self.choices or self.allow_other or answer in self.choices


class HumanReviewForm:
    """`HitlWorkflow` callbacks that render every pending call in one form.

    ``review`` reads the application's function-call payload. It runs on the
    trusted persisted calls, both when the form is published and when an answer
    is validated, and raises ``ValueError`` for a payload it cannot present.
    """

    def __init__(self, review: Callable[[ResponseFunctionToolCall], HumanReview]):
        self._review = review

    def prompt(self, calls: Sequence[ResponseFunctionToolCall]) -> str:
        """Return the message text that accompanies the form."""
        reviews = [self._review(call) for call in calls]
        if len(reviews) == 1:
            return reviews[0].prompt
        prompts = "\n\n".join(
            f"{index}. {review.prompt}" for index, review in enumerate(reviews, 1)
        )
        return f"Human review is required for {len(reviews)} requests.\n\n{prompts}"

    def props(self, calls: Sequence[ResponseFunctionToolCall]) -> dict[str, object]:
        """Return the element props for the complete call batch."""
        return {
            "reviews": [
                {
                    "prompt": review.prompt,
                    "choices": list(review.choices),
                    "allow_other": review.allow_other or not review.choices,
                }
                for review in map(self._review, calls)
            ]
        }

    def validate_outputs(
        self,
        calls: Sequence[ResponseFunctionToolCall],
        outputs: Sequence[str],
    ) -> tuple[str, ...]:
        """Return trimmed answers after checking them against the trusted calls."""
        if len(outputs) != len(calls):
            raise ValueError("Every human-review request requires one response.")
        answers = []
        for call, output in zip(calls, outputs, strict=True):
            if not isinstance(output, str) or not (answer := output.strip()):
                raise ValueError("Human-review responses must be non-empty strings.")
            if not self._review(call).accepts(answer):
                raise ValueError("A human-review response is not an allowed choice.")
            answers.append(answer)
        return tuple(answers)


__all__ = ["HUMAN_REVIEW_ELEMENT_NAME", "HumanReview", "HumanReviewForm"]
