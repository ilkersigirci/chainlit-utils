import inspect
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import chainlit.mcp
import chainlit.server
import pytest
import uvicorn
from chainlit.context import init_http_context
from chainlit.user_session import user_sessions
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, Tool
from openai.types.responses import ResponseFunctionToolCall

from chainlit_utils import mcp


class StepDouble:
    def __init__(self, **_: object) -> None:
        self.input: object = None
        self.output: object = None

    async def __aenter__(self) -> "StepDouble":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


def _call(name: str, arguments: str = "{}") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id="fc-1",
        call_id="call-1",
        name=name,
        arguments=arguments,
        status="completed",
        type="function_call",
    )


@asynccontextmanager
async def _report_server() -> AsyncIterator[tuple[str, list[tuple[str, str]]]]:
    """Serve one tool over streamable HTTP and record each request's credential."""
    server = FastMCP("reports")

    @server.tool()
    def count_users() -> int:
        """Count the users."""
        return 12

    app = server.streamable_http_app()
    requests: list[tuple[str, str]] = []

    async def recording_app(scope, receive, send) -> None:
        if scope["type"] == "http":
            credential = dict(scope["headers"]).get(b"authorization", b"").decode()
            requests.append((scope["path"], credential))
        await app(scope, receive, send)

    http = uvicorn.Server(
        uvicorn.Config(recording_app, host="127.0.0.1", port=0, log_level="warning")
    )
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(http.serve)
        with anyio.fail_after(10):
            while not http.started:
                await anyio.sleep(0.01)
        port = http.servers[0].sockets[0].getsockname()[1]
        try:
            yield f"http://127.0.0.1:{port}", requests
        finally:
            http.should_exit = True


@pytest.fixture
async def chainlit_context():
    context = init_http_context()
    try:
        yield context
    finally:
        user_sessions.pop(context.session.id, None)


def test_server_config_uses_the_supplied_identity_endpoint_and_headers() -> None:
    tools = mcp.McpTools("company-tools")

    server = tools.server(
        "https://mcp.example/tools",
        headers={"X-API-Key": "secret"},
    )

    assert (server.name, server.url) == (
        "company-tools",
        "https://mcp.example/tools",
    )
    assert server.headers == {"X-API-Key": "secret"}


def test_a_static_api_key_is_a_bearer_header_for_the_whole_session() -> None:
    server = mcp.McpTools("company-tools").server(
        "https://mcp.example/tools", headers={"X-Team": "reports"}, api_key="secret"
    )

    assert server.headers == {"X-Team": "reports", "Authorization": "Bearer secret"}


async def _token() -> str:
    return "token"


@pytest.mark.parametrize("api_key", ["secret", _token], ids=["static", "callable"])
def test_api_key_and_an_authorization_header_are_exclusive(api_key) -> None:
    with pytest.raises(ValueError, match="api_key or headers"):
        mcp.McpTools("company-tools").server(
            "https://mcp.example/tools",
            headers={"authorization": "Bearer other"},
            api_key=api_key,
        )


def test_server_config_allows_an_unauthenticated_endpoint() -> None:
    server = mcp.McpTools("public-tools").server("https://mcp.example")

    assert server.headers == {}


def test_server_name_must_be_a_nonempty_string() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        mcp.McpTools(0)  # type: ignore[arg-type]


async def test_session_discovers_executes_and_disconnects_advertised_tools(
    chainlit_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advertised = Tool(
        name="database_report",
        description="Run a report.",
        inputSchema={"type": "object", "properties": {}},
    )
    session = AsyncMock()
    session.list_tools.return_value = SimpleNamespace(tools=[advertised])
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text='[{"count":12}]')]
    )
    monkeypatch.setattr(mcp.cl, "Step", StepDouble)
    tools = mcp.McpTools("company-tools")

    await tools.connect(SimpleNamespace(name="company-tools"), session)
    assert [tool["name"] for tool in tools.response_tools()] == [advertised.name]

    output = await tools.execute(_call(advertised.name))
    session.call_tool.assert_awaited_once_with(advertised.name, {})
    assert output["call_id"] == "call-1"
    assert json.loads(output["output"])["content"][0]["text"] == '[{"count":12}]'

    replacement = AsyncMock()
    replacement.list_tools.return_value = SimpleNamespace(tools=[advertised])
    await tools.connect(SimpleNamespace(name="company-tools"), replacement)
    await tools.disconnect("company-tools", session)
    assert tools.response_tools()
    await tools.disconnect("company-tools", replacement)
    assert tools.response_tools() == []


@pytest.mark.parametrize("arguments", ["[]", "not-json"])
async def test_execution_requires_json_object_arguments(
    chainlit_context,
    arguments: str,
) -> None:
    tool = Tool(name="lookup", inputSchema={"type": "object"})
    session = AsyncMock()
    session.list_tools.return_value = SimpleNamespace(tools=[tool])
    tools = mcp.McpTools("company-tools")
    await tools.connect(SimpleNamespace(name="company-tools"), session)

    with pytest.raises(ValueError, match="arguments"):
        await tools.execute(_call("lookup", arguments))
    session.call_tool.assert_not_awaited()


# Chainlit 2.12 connects with the deprecated SDK client this test mirrors.
@pytest.mark.filterwarnings("ignore:Use `streamable_http_client`:DeprecationWarning")
async def test_native_connection_resolves_a_callable_api_key_for_each_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp, "_api_keys", {})
    monkeypatch.setattr(
        chainlit.mcp,
        "make_mcp_http_client_factory",
        chainlit.mcp.make_mcp_http_client_factory,
    )
    issued: list[str] = []

    async def api_key() -> str:
        # A refreshed delegated token differs from the one before it.
        issued.append(f"token-{len(issued) + 1}")
        return issued[-1]

    async with _report_server() as (root, requests):
        server = mcp.McpTools("reports").server(f"{root}/mcp", api_key=api_key)
        # Chainlit's /mcp endpoint builds its connection like this.
        factory = chainlit.mcp.make_mcp_http_client_factory(lambda _url: None)
        async with (
            streamablehttp_client(
                server.url, headers=server.headers, httpx_client_factory=factory
            ) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool("count_users", {})
        async with factory() as client:
            await client.get(f"{root}/other")

    assert result.content[0].text == "12"
    # Initialization and the call each resolved a new token.
    assert len(issued) >= 2
    assert sorted(credential for path, credential in requests if path == "/mcp") == (
        sorted(f"Bearer {token}" for token in issued)
    )
    assert ("/other", "") in requests


def test_chainlit_builds_mcp_clients_in_the_session_context_at_connect_time() -> None:
    # The API key wrapper depends on both; recheck it if a Chainlit upgrade
    # fails this test.
    source = inspect.getsource(chainlit.server.connect_mcp)

    assert "make_mcp_http_client_factory," in source
    assert source.index("init_ws_context(session)") < source.index(
        "asyncio.create_task("
    )
