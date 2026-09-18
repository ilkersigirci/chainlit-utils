import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from chainlit.context import init_http_context
from chainlit.user_session import user_sessions
from mcp.types import CallToolResult, TextContent, Tool
from openai.types.responses import ResponseFunctionToolCall

from chainlit_utils import mcp


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
    class StepDouble:
        def __init__(self, **_: object) -> None:
            self.input: object = None
            self.output: object = None

        async def __aenter__(self) -> "StepDouble":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

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

    output = await tools.execute(
        ResponseFunctionToolCall(
            id="fc-1",
            call_id="call-1",
            name=advertised.name,
            arguments="{}",
            status="completed",
            type="function_call",
        )
    )
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
        await tools.execute(
            ResponseFunctionToolCall(
                id="fc-1",
                call_id="call-1",
                name="lookup",
                arguments=arguments,
                status="completed",
                type="function_call",
            )
        )
    session.call_tool.assert_not_awaited()
