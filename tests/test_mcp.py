"""MCPTools against the real MCP SDK, connected in-process to a real server."""

import sys

import pytest

mcp = pytest.importorskip("mcp")
if not hasattr(mcp, "Client"):
    pytest.skip("needs MCP SDK 2 or newer", allow_module_level=True)

from mcp.server.mcpserver import Image, MCPServer  # noqa: E402

from agentloom import Agent, Approval, ConfigurationError  # noqa: E402
from agentloom.mcp import MCPTools  # noqa: E402
from agentloom.testing import ScriptedProvider, call  # noqa: E402


def make_server() -> MCPServer:
    server = MCPServer("demo")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool()
    def profile(user: str) -> dict:
        """Look up a user's profile."""
        return {"user": user, "plan": "pro"}

    @server.tool()
    def fail(reason: str) -> str:
        """Always fails."""
        raise ValueError(reason)

    @server.tool(name="files.delete")
    def delete(path: str) -> str:
        """Delete a file."""
        return f"deleted {path}"

    @server.tool()
    def logo() -> Image:
        """Return an image."""
        return Image(data=b"\x89PNG", format="png")

    return server


def last_results(provider: ScriptedProvider):
    return provider.requests[-1].messages[-1].tool_results


async def test_server_tools_become_agent_tools():
    async with MCPTools(make_server()) as tools:
        by_name = {t.name: t for t in tools}
        assert list(by_name) == ["add", "profile", "fail", "files_delete", "logo"]
        assert by_name["add"].description == "Add two numbers."
        assert by_name["add"].input_schema["required"] == ["a", "b"]

        provider = ScriptedProvider([[call("add", a=2, b=3), call("profile", user="ada")], "Done."])
        result = await Agent(provider, tools=tools).run("go")

    assert result.ok
    assert last_results(provider)[0].content == "5"
    assert '"plan": "pro"' in last_results(provider)[1].content
    assert len(tools) == 0  # the tools are gone once the connection is closed


async def test_failures_reach_the_model_as_error_results():
    async with MCPTools(make_server()) as tools:
        provider = ScriptedProvider(
            [[call("fail", reason="boom"), call("add", a="x", b=1), call("logo")], "ok"]
        )
        await Agent(provider, tools=tools).run("go")

    failed, invalid, image = last_results(provider)
    assert failed.is_error and "fail" in failed.content
    # Rejected by agentloom's own validation, before the server is called.
    assert invalid.is_error and "$.a: expected integer" in invalid.content
    assert not image.is_error and image.content == "[image omitted: image/png]"


async def test_prefix_include_and_approval():
    asked = []

    async def approver(tool_call, ctx):
        asked.append(tool_call.name)
        return Approval(False, "read-only session")

    options = {
        "prefix": "fs",
        "include": ["add", "files.delete"],
        "requires_approval": ["files.delete"],
        "timeout": 5,
    }
    async with MCPTools(make_server(), **options) as tools:
        assert [(t.name, t.requires_approval, t.timeout) for t in tools] == [
            ("fs_add", False, 5),
            ("fs_files_delete", True, 5),
        ]
        provider = ScriptedProvider(
            [[call("fs_add", a=1, b=1), call("fs_files_delete", path="/etc")], "ok"]
        )
        await Agent(provider, tools=tools, approval=approver).run("go")

    added, deleted = last_results(provider)
    assert added.content == "2" and asked == ["fs_files_delete"]
    assert deleted.is_error and "not approved: read-only session" in deleted.content


async def test_calls_after_close_fail_cleanly():
    async with MCPTools(make_server()) as tools:
        kept = tools.tools
    provider = ScriptedProvider([call("add", a=1, b=1), "ok"])

    await Agent(provider, tools=kept).run("go")

    outcome = last_results(provider)[0]
    assert outcome.is_error and "connection to the MCP server is closed" in outcome.content


async def test_colliding_names_are_rejected():
    server = MCPServer("clash")

    @server.tool(name="a.b")
    def first() -> str:
        """First."""
        return ""

    @server.tool(name="a_b")
    def second() -> str:
        """Second."""
        return ""

    with pytest.raises(ConfigurationError, match="both map to the name 'a_b'"):
        async with MCPTools(server):
            pass


def test_stdio_builds_server_parameters():
    tools = MCPTools.stdio("npx", "-y", "some-server", env={"TOKEN": "t"}, cwd=".", prefix="x")

    parameters = tools._server
    assert (parameters.command, parameters.args) == ("npx", ["-y", "some-server"])
    assert parameters.env == {"TOKEN": "t"} and tools._prefix == "x"


STDIO_SERVER = '''
from mcp.server.mcpserver import MCPServer

server = MCPServer("stdio-demo")


@server.tool()
def shout(text: str) -> str:
    """Upper-case some text."""
    return text.upper()


server.run("stdio")
'''


async def test_stdio_subprocess_round_trip(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(STDIO_SERVER, encoding="utf-8")

    async with MCPTools.stdio(sys.executable, str(script)) as tools:
        provider = ScriptedProvider([call("shout", text="hi"), "ok"])
        await Agent(provider, tools=tools).run("go")

    assert last_results(provider)[0].content == "HI"
