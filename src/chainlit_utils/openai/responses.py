"""Project and render OpenAI Responses in Chainlit applications."""

from collections.abc import Mapping, Sequence
from typing import Any

import chainlit as cl
from chainlit.element import Element
from openai.types.responses import Response


class CommentaryTaskList:
    """Render streamed commentary as one native Chainlit task list."""

    def __init__(self) -> None:
        self._task_list: cl.TaskList | None = None
        self._active_task: cl.Task | None = None

    async def add(self, content: str) -> None:
        """Complete the prior task and append the latest status as running."""
        if not content:
            return
        if self._task_list is None:
            self._task_list = cl.TaskList()
        if self._active_task is not None:
            self._active_task.status = cl.TaskStatus.DONE

        task = cl.Task(title=content, status=cl.TaskStatus.RUNNING)
        await self._task_list.add_task(task)
        self._task_list.status = "Running..."
        self._active_task = task
        await self._task_list.send()

    async def complete(self) -> None:
        """Mark the task list complete after the Responses loop succeeds."""
        if self._task_list is None:
            return
        if self._active_task is not None:
            self._active_task.status = cl.TaskStatus.DONE
            self._active_task = None
        self._task_list.status = "Done"
        await self._task_list.send()

    async def stop(self) -> None:
        """Mark the active task as failed when the Responses loop stops early."""
        if self._task_list is None or self._active_task is None:
            return
        self._active_task.status = cl.TaskStatus.FAILED
        self._active_task = None
        self._task_list.status = "Stopped"
        await self._task_list.send()


def response_input(messages: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    """Convert Chainlit's text transcript to Responses message items."""
    items = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            continue
        item = {"role": role, "content": content}
        if role == "assistant":
            item["phase"] = message.get("phase") or "final_answer"
        items.append(item)
    return items


def final_answer(response: Response) -> str:
    """Return durable final-answer text without concatenating commentary."""
    parts = []
    for item in response.output:
        if item.type != "message" or item.phase == "commentary":
            continue
        parts.extend(
            part.text if part.type == "output_text" else part.refusal
            for part in item.content
        )
    return "".join(parts)


def citation_elements(response: Response) -> list[Element]:
    """Make final-answer URL citations clickable without changing the transcript."""
    elements: list[Element] = []
    for item in response.output:
        if item.type != "message" or item.phase == "commentary":
            continue
        for part in item.content:
            if part.type != "output_text":
                continue
            for annotation in part.annotations:
                if annotation.type != "url_citation":
                    continue
                stop = annotation.end_index + 1
                if 0 <= annotation.start_index < stop <= len(part.text):
                    elements.append(
                        cl.Text(
                            name=part.text[annotation.start_index : stop],
                            content=f"[Open source](<{annotation.url}>)",
                            display="side",
                        )
                    )
    return elements


def raise_for_response(response: Response) -> None:
    """Only completed Responses may be rendered or continued as successful."""
    if response.status == "completed":
        return
    if response.status == "incomplete":
        reason = response.incomplete_details
        raise RuntimeError(
            f"Response incomplete: {reason.reason if reason else 'unknown reason'}."
        )
    detail = response.error
    raise RuntimeError(detail.message if detail is not None else "Response failed.")


__all__ = [
    "CommentaryTaskList",
    "citation_elements",
    "final_answer",
    "raise_for_response",
    "response_input",
]
