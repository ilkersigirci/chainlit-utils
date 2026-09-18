"""Durable OpenAI Responses continuations for human-in-the-loop workflows."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chainlit_utils.openai.tools import function_call_output

HITL_LEDGER_SCHEMA_VERSION = 2


class InvalidHitlLedgerError(ValueError):
    """A persisted HITL continuation is unsafe to resume."""


@dataclass(frozen=True, slots=True)
class HitlContinuation:
    """The validated Responses values needed to resume one complete call batch."""

    model_id: str
    response_id: str
    function_calls: tuple[ResponseFunctionToolCall, ...]


class _LedgerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _DirectCaller(_LedgerModel):
    type: Literal["direct"]


class _ProgramCaller(_LedgerModel):
    caller_id: str = Field(min_length=1)
    type: Literal["program"]


class _PersistedFunctionCall(_LedgerModel):
    arguments: str
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: Literal["function_call"]
    id: str | None = Field(default=None, min_length=1)
    async_: bool | None = Field(default=None, alias="async")
    caller: _DirectCaller | _ProgramCaller | None = None
    namespace: str | None = Field(default=None, min_length=1)
    status: Literal["completed"] | None = None


class _PendingLedger(_LedgerModel):
    schema_version: Literal[2]
    status: Literal["pending"]
    model_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    function_calls: list[_PersistedFunctionCall] = Field(min_length=1)


class _CompletedLedger(_LedgerModel):
    schema_version: Literal[2]
    status: Literal["completed"]


class HitlLedgerCodec:
    """Validate and serialize one application's HITL function calls."""

    def __init__(self, tool_name: str) -> None:
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("HITL tool name must be a non-empty string.")
        self.tool_name = tool_name

    def continuation(
        self,
        *,
        model_id: str,
        response_id: str,
        function_calls: Sequence[ResponseFunctionToolCall],
    ) -> HitlContinuation:
        """Validate SDK output before it becomes client-owned durable state."""
        return self._decode_pending(
            {
                "schema_version": HITL_LEDGER_SCHEMA_VERSION,
                "status": "pending",
                "model_id": model_id,
                "response_id": response_id,
                "function_calls": [
                    call.model_dump(
                        mode="json",
                        by_alias=True,
                        exclude_none=True,
                    )
                    for call in function_calls
                ],
            }
        )

    def pending_metadata(
        self,
        continuation: HitlContinuation,
    ) -> dict[str, object]:
        """Encode a pending continuation for durable message metadata."""
        ledger = self._validate_pending(
            {
                "schema_version": HITL_LEDGER_SCHEMA_VERSION,
                "status": "pending",
                "model_id": continuation.model_id,
                "response_id": continuation.response_id,
                "function_calls": [
                    call.model_dump(
                        mode="json",
                        by_alias=True,
                        exclude_none=True,
                    )
                    for call in continuation.function_calls
                ],
            }
        )
        return ledger.model_dump(mode="json", by_alias=True, exclude_none=True)

    def completed_metadata(self) -> dict[str, object]:
        """Encode the terminal marker that prevents older pending replay."""
        return _CompletedLedger(
            schema_version=HITL_LEDGER_SCHEMA_VERSION,
            status="completed",
        ).model_dump(mode="json")

    def decode(self, raw_ledger: object) -> HitlContinuation | None:
        """Decode one strict ledger value; completed values return ``None``."""
        if not isinstance(raw_ledger, dict):
            raise InvalidHitlLedgerError("HITL ledger metadata is not an object.")
        if raw_ledger.get("schema_version") != HITL_LEDGER_SCHEMA_VERSION:
            raise InvalidHitlLedgerError("HITL ledger schema is unsupported.")

        status = raw_ledger.get("status")
        if status == "completed":
            try:
                _CompletedLedger.model_validate(raw_ledger)
            except ValidationError as exc:
                raise InvalidHitlLedgerError(
                    "Completed HITL ledger is invalid."
                ) from exc
            return None
        if status != "pending":
            raise InvalidHitlLedgerError("HITL ledger status is invalid.")
        return self._decode_pending(raw_ledger)

    def _decode_pending(self, raw_ledger: object) -> HitlContinuation:
        ledger = self._validate_pending(raw_ledger)
        calls = tuple(
            ResponseFunctionToolCall.model_validate(
                call.model_dump(
                    mode="python",
                    by_alias=True,
                    exclude_none=True,
                )
            )
            for call in ledger.function_calls
        )
        unexpected_tools = sorted(
            {call.name for call in calls if call.name != self.tool_name}
        )
        if unexpected_tools:
            raise InvalidHitlLedgerError(
                f"HITL ledger contains unsupported functions: {unexpected_tools!r}."
            )
        call_ids = {call.call_id for call in calls}
        if len(call_ids) != len(calls):
            raise InvalidHitlLedgerError("HITL ledger call IDs must be unique.")
        return HitlContinuation(
            model_id=ledger.model_id,
            response_id=ledger.response_id,
            function_calls=calls,
        )

    @staticmethod
    def _validate_pending(raw_ledger: object) -> _PendingLedger:
        try:
            return _PendingLedger.model_validate(raw_ledger)
        except ValidationError as exc:
            raise InvalidHitlLedgerError(
                "HITL ledger continuation is invalid."
            ) from exc


def function_call_outputs(
    continuation: HitlContinuation,
    outputs: Sequence[str],
) -> list[dict[str, Any]]:
    """Build one complete Responses function-output batch in call order."""
    if len(outputs) != len(continuation.function_calls):
        raise ValueError("Every HITL function call requires one output.")
    if any(not isinstance(output, str) for output in outputs):
        raise TypeError("HITL function outputs must be strings.")
    return [
        function_call_output(call, output)
        for call, output in zip(continuation.function_calls, outputs, strict=True)
    ]


__all__ = [
    "HITL_LEDGER_SCHEMA_VERSION",
    "HitlContinuation",
    "HitlLedgerCodec",
    "InvalidHitlLedgerError",
    "function_call_outputs",
]
