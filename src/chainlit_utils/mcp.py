"""Bridge Chainlit-managed MCP sessions to OpenAI Responses tools."""

import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import chainlit as cl
import chainlit.mcp
import httpx
from chainlit.config import StreamableHttpMcpServer
from chainlit.mcp import McpConnection, McpHttpClientFactory
from mcp import ClientSession
from mcp.types import Tool
from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall

from chainlit_utils.openai.tools import function_call_output

# Resolved for each request to the server URL, as the OpenAI SDK resolves a
# callable api_key.
_api_keys: dict[httpx.URL, Callable[[], Awaitable[str]]] = {}


@dataclass(frozen=True)
class _ConnectedServer:
    session: ClientSession
    tools: tuple[Tool, ...]


class _ApiKeyAuth(httpx.Auth):
    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        api_key = _api_keys.get(request.url)
        if api_key is not None:
            request.headers["Authorization"] = f"Bearer {await api_key()}"
        yield request


_chainlit_client_factory = chainlit.mcp.make_mcp_http_client_factory


def _client_factory_with_api_keys(*args: Any, **kwargs: Any) -> McpHttpClientFactory:
    """
    Wrap Chainlit's MCP HTTP client factory to resolve callable API keys.

    Chainlit has no hook for per-request MCP headers, but it builds every
    connection's client through this factory at connect time, in a task that
    runs in the connecting user's session context.
    """
    factory = _chainlit_client_factory(*args, **kwargs)

    def with_api_keys(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return factory(headers=headers, timeout=timeout, auth=auth or _ApiKeyAuth())

    return with_api_keys


class McpTools:
    """Expose tools from one Chainlit MCP connection to Responses requests."""

    def __init__(self, server_name: str, *, session_key: str | None = None) -> None:
        if not isinstance(server_name, str) or not server_name:
            raise ValueError("MCP server name cannot be empty.")
        self.server_name = server_name
        self.session_key = session_key or f"chainlit_utils.mcp.{server_name}"

    def server(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        api_key: str | Callable[[], Awaitable[str]] | None = None,
    ) -> StreamableHttpMcpServer:
        """
        Build a Chainlit streamable-HTTP server configuration.

        ``api_key`` is sent as a bearer token, like the OpenAI SDK's: a string
        for the whole session, or an async callable that returns the token for
        each request, such as the signed-in user's refreshed delegated token.
        """
        server_headers = dict(headers or {})
        if api_key is not None and any(
            name.lower() == "authorization" for name in server_headers
        ):
            raise ValueError("Pass the MCP credential as api_key or headers.")
        if isinstance(api_key, str):
            server_headers["Authorization"] = f"Bearer {api_key}"
        elif api_key is not None:
            _api_keys[httpx.URL(url)] = api_key
            chainlit.mcp.make_mcp_http_client_factory = _client_factory_with_api_keys  # ty: ignore[invalid-assignment] - Replacing Chainlit's function is the intent.
        return StreamableHttpMcpServer(
            name=self.server_name,
            type="streamable-http",
            url=url,
            headers=server_headers,
        )

    async def connect(
        self,
        connection: McpConnection,
        session: ClientSession,
    ) -> None:
        """Discover tools after Chainlit connects to the configured server."""
        if connection.name != self.server_name:
            return
        discovered = await session.list_tools()
        cl.user_session.set(
            self.session_key,
            _ConnectedServer(session=session, tools=tuple(discovered.tools)),
        )

    async def disconnect(self, name: str, session: ClientSession) -> None:
        """Forget tools when the active MCP session disconnects."""
        server = self._connected_server()
        if name != self.server_name or server is None or server.session is not session:
            return
        cl.user_session.set(self.session_key, None)

    def response_tools(self) -> list[FunctionToolParam]:
        """Return the connected MCP tools in Responses API format."""
        server = self._connected_server()
        if server is None:
            return []

        response_tools: list[FunctionToolParam] = []
        for tool in server.tools:
            response_tool: FunctionToolParam = {
                "type": "function",
                "name": tool.name,
                "parameters": tool.inputSchema,
                "strict": False,
            }
            if tool.description is not None:
                response_tool["description"] = tool.description
            response_tools.append(response_tool)
        return response_tools

    async def execute(self, call: ResponseFunctionToolCall) -> dict[str, object]:
        """Execute an advertised tool and return its Responses function output."""
        server = self._connected_server()
        if server is None or not any(tool.name == call.name for tool in server.tools):
            raise ValueError(f"MCP tool is not connected: {call.name}")
        try:
            arguments = json.loads(call.arguments)
        except ValueError as exc:
            raise ValueError(
                f"MCP tool call contains invalid arguments: {call.name}"
            ) from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"MCP tool arguments must be an object: {call.name}")

        async with cl.Step(name=call.name, type="tool") as step:
            step.input = arguments
            result = await server.session.call_tool(call.name, arguments)
            output = result.model_dump_json(by_alias=True, exclude_none=True)
            step.output = output
        return function_call_output(call, output)

    def _connected_server(self) -> _ConnectedServer | None:
        server = cl.user_session.get(self.session_key)
        return server if isinstance(server, _ConnectedServer) else None


__all__ = ["McpTools"]
