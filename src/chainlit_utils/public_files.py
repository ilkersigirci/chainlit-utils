"""Serve the package's Chainlit custom elements and browser scripts."""

from pathlib import Path

from chainlit.config import config
from chainlit.server import app
from starlette.responses import FileResponse
from starlette.routing import Route

PUBLIC_DIR = Path(__file__).with_name("public")
# Scripts share Chainlit's /public namespace, so they get their own prefix.
SCRIPTS_PATH = "/public/chainlit-utils"


def serve_public_files() -> None:
    """Serve the bundled elements and scripts from Chainlit's own server.

    Chainlit 2.12 reads custom elements only from
    ``public/elements/{name}.jsx`` under the application root and has no
    setting for another directory. These routes are placed ahead of Chainlit's
    public-file route, so a package element replaces an application file of
    the same name while every other application file is still served. Because
    they belong to Chainlit's app, they work for ``chainlit run`` and under
    ``mount_chainlit`` at any path. Call this once at application startup.

    Scripts are served under ``/public/chainlit-utils/``; Chainlit's
    ``custom_js`` setting takes one URL, so the application selects the script.
    """
    root = config.run.root_path
    routes = [
        _file_route(f"{root}/public/elements/{file.name}", file)
        for file in sorted((PUBLIC_DIR / "elements").glob("*.jsx"))
    ]
    routes += [
        _file_route(f"{root}{SCRIPTS_PATH}/{file.name}", file)
        for file in sorted(PUBLIC_DIR.glob("*.js"))
    ]
    app.router.routes[:0] = routes


def _file_route(path: str, file: Path) -> Route:
    async def endpoint(_request: object) -> FileResponse:
        return FileResponse(file)

    return Route(path, endpoint, methods=["GET"], include_in_schema=False)


__all__ = ["SCRIPTS_PATH", "serve_public_files"]
