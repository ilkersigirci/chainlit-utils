from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from chainlit.config import public_dir
from chainlit.server import app as chainlit_app
from fastapi import FastAPI

from chainlit_utils import public_files


@pytest.fixture
def app_elements(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    # Restore Chainlit's own routes after the test.
    monkeypatch.setattr(chainlit_app.router, "routes", list(chainlit_app.router.routes))
    elements = Path(public_dir) / "elements"
    elements.mkdir(parents=True, exist_ok=True)
    files = {"AppElement.jsx": "// app element", "SpeechButton.jsx": "// app copy"}
    for name, content in files.items():
        (elements / name).write_text(content)
    try:
        yield elements
    finally:
        for name in files:
            (elements / name).unlink()


@pytest.mark.parametrize("mount_path", ["", "/chat"], ids=["root", "mounted"])
async def test_package_files_are_served_next_to_application_files(
    app_elements: Path,
    mount_path: str,
) -> None:
    public_files.serve_public_files()
    parent = FastAPI()
    parent.mount(mount_path, chainlit_app)
    package = public_files.PUBLIC_DIR

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=parent), base_url="http://chainlit"
    ) as client:

        async def get(path: str) -> httpx.Response:
            return await client.get(f"{mount_path}{path}")

        speech_button = await get("/public/elements/SpeechButton.jsx")
        dictation = await get("/public/chainlit-utils/dictation.js")
        app_element = await get("/public/elements/AppElement.jsx")
        missing = await get("/public/elements/Missing.jsx")

    assert speech_button.text == (package / "elements/SpeechButton.jsx").read_text()
    assert dictation.text == (package / "dictation.js").read_text()
    assert app_element.text == "// app element"
    assert missing.status_code == 404
