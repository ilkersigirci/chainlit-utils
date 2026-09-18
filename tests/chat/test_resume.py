from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from chainlit_utils.chat import resume


async def test_schedule_after_hydration_tracks_the_chainlit_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = AsyncMock()
    session = SimpleNamespace(current_task=None)
    monkeypatch.setattr(resume.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(
        resume,
        "chainlit_context",
        SimpleNamespace(session=session),
    )

    task = resume.schedule_after_thread_hydration(callback)
    await task

    assert session.current_task is task
    callback.assert_awaited_once_with()


def test_reuse_persisted_step_copies_chainlit_identity() -> None:
    source = Mock(
        id="persisted",
        parent_id="parent",
        created_at="2026-08-10T12:00:00Z",
        metadata={"state": "pending"},
    )
    target = SimpleNamespace(
        id="transient",
        parent_id=None,
        created_at=None,
        metadata=None,
        persisted=False,
    )

    resume.reuse_persisted_step(target, source)

    assert target.id == "persisted"
    assert target.parent_id == "parent"
    assert target.created_at == source.created_at
    assert target.metadata is source.metadata
    assert target.persisted is True
