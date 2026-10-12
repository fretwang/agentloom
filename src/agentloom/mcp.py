"""Use the tools of an MCP server (``pip install agentloom[mcp]``).

    async with MCPTools.stdio("npx", "-y", "@modelcontextprotocol/server-filesystem", ".") as fs:
        agent = Agent(provider, tools=fs)
        await agent.run("What is in this directory?")

Each server tool becomes an ordinary :class:`~agentloom.Tool`, so limits,
approval, hooks and tracing apply to it like to any other tool.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterator, Mapping
from typing import Any

from .errors import ConfigurationError, ToolError
from .tools import Tool, ToolContext

_INVALID = re.compile(r"[^a-zA-Z0-9_-]")


class MCPTools:
    """A connection to one MCP server, iterable as its tools.

    Enter it with ``async with``; the tools are valid until the block exits.

    Args:
        server: A Streamable HTTP URL, an ``mcp.StdioServerParameters``, or
            anything else ``mcp.Client`` accepts. :meth:`stdio` builds the
            parameters for a local command.
        prefix: Put ``<prefix>_`` before every tool name. Use it when two
            servers have tools with the same name.
        include: Expose only the server tools with these names.
        requires_approval: ``True`` routes every call through the agent's
            approval handler; a collection of server tool names does so for
            those tools only. A server's tools are code you did not write:
            gate the ones that change things.
        timeout: Seconds allowed per call; ``None`` uses the agent's
            ``RunLimits.tool_timeout``.
    """

    def __init__(
        self,
        server: Any,
        *,
        prefix: str | None = None,
        include: Collection[str] | None = None,
        requires_approval: bool | Collection[str] = False,
        timeout: float | None = None,
    ) -> None:
        try:
            import mcp
        except ImportError as exc:
            raise ConfigurationError(
                "MCPTools needs the MCP SDK: pip install 'agentloom[mcp]'"
            ) from exc
        if not hasattr(mcp, "Client"):
            raise ConfigurationError("MCPTools needs version 2 or newer of the MCP SDK")
        self._sdk: Any = mcp
        self._server = server
        self._prefix = prefix
        self._include = include
        self._requires_approval = requires_approval
        self._timeout = timeout
        self._client: Any = None
        self._tools: list[Tool] = []

    @classmethod
    def stdio(
        cls,
        command: str,
        *args: str,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        **options: Any,
    ) -> MCPTools:
        """Launch ``command`` as a subprocess and talk to it over stdin/stdout."""
        try:
            from mcp import StdioServerParameters
        except ImportError as exc:
            raise ConfigurationError(
                "MCPTools needs the MCP SDK: pip install 'agentloom[mcp]'"
            ) from exc
        parameters = StdioServerParameters(
            command=command, args=list(args), env=dict(env) if env is not None else None, cwd=cwd
        )
        return cls(parameters, **options)

    async def __aenter__(self) -> MCPTools:
        client = self._sdk.Client(self._server)
        await client.__aenter__()
        self._client = client
        try:
            self._tools = await self._load()
        except BaseException:
            self._client = None
            await client.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        client, self._client, self._tools = self._client, None, []
        if client is not None:
            await client.__aexit__(*exc_info)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    async def _load(self) -> list[Tool]:
        listed: list[Any] = []
        cursor: str | None = None
        while True:
            page = await self._client.list_tools(cursor=cursor)
            listed += page.tools
            cursor = page.next_cursor
            if not cursor:
                break

        tools: dict[str, Tool] = {}
        for remote in listed:
            if self._include is not None and remote.name not in self._include:
                continue
            full = f"{self._prefix}_{remote.name}" if self._prefix else remote.name
            name = _INVALID.sub("_", full)[:64]
            if name in tools:
                raise ConfigurationError(
                    f"MCP server exposes two tools that both map to the name {name!r}"
                )
            gate = self._requires_approval
            tools[name] = Tool(
                name=name,
                description=remote.description or remote.title or f"MCP tool {remote.name}.",
                input_schema=dict(remote.input_schema or {"type": "object"}),
                handler=self._handler(remote.name),
                requires_approval=gate if isinstance(gate, bool) else remote.name in gate,
                timeout=self._timeout,
            )
        return list(tools.values())

    def _handler(self, remote_name: str) -> Any:
        async def handler(arguments: Mapping[str, Any], ctx: ToolContext) -> str:
            client = self._client
            if client is None:
                raise ToolError("the connection to the MCP server is closed")
            try:
                result = await client.call_tool(remote_name, dict(arguments))
            except self._sdk.MCPError as exc:
                raise ToolError(f"MCP error: {exc}") from exc
            text = _render(result)
            if result.is_error:
                raise ToolError(text or "The tool reported a failure.")
            return text

        return handler


def _render(result: Any) -> str:
    """Flatten an MCP tool result into text for the model."""
    parts: list[str] = []
    for item in result.content or ():
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(item.text)
        elif kind == "resource" and getattr(item.resource, "text", None):
            parts.append(item.resource.text)
        elif kind == "resource_link":
            parts.append(f"[resource: {item.uri}]")
        else:
            mime = getattr(item, "mime_type", None)
            parts.append(f"[{kind or 'content'} omitted{f': {mime}' if mime else ''}]")
    if not parts and result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False, default=str)
    return "\n".join(parts)
