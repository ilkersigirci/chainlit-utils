import os
import tempfile
from pathlib import Path

import pytest

_chainlit_app_root = tempfile.TemporaryDirectory(prefix="chainlit-utils-tests-")
os.environ["CHAINLIT_APP_ROOT"] = _chainlit_app_root.name


@pytest.fixture(autouse=True)
def chainlit_app_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def runtime_settings() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "use_history": {
                    "type": "boolean",
                    "title": "Use conversation history",
                    "default": True,
                },
                "mode": {
                    "type": "string",
                    "title": "Mode",
                    "enum": ["brief", "detailed"],
                    "default": "brief",
                },
                "assistant_name": {
                    "type": "string",
                    "title": "Assistant name",
                    "minLength": 1,
                    "default": "Helper",
                },
            },
        },
        {
            "use_history": True,
            "mode": "brief",
            "assistant_name": "Helper",
        },
    )
