"""Bridge Chainlit-managed MCP sessions to OpenAI Responses tools."""

import json
from collections.abc import Mapping
from dataclasses import dataclass

import chainlit as cl
from chainlit.config import StreamableHttpMcpServer
from chainlit.mcp import McpConnection
from mcp import ClientSession
from mcp.types import Tool
from openai.types.responses import FunctionToolParam, ResponseFunctionToolCall

from chainlit_utils.openai.tools import function_call_output


@dataclass(frozen=True)
class _ConnectedServer:
    session: ClientSession
    tools: tuple[Tool, ...]


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
    ) -> StreamableHttpMcpServer:
        """Build a Chainlit streamable-HTTP server configuration."""
        return StreamableHttpMcpServer(
            name=self.server_name,
            type="streamable-http",
            url=url,
            headers=dict(headers or {}),
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
