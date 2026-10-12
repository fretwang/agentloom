"""Use the tools of an MCP server.

Starts examples/mcp_demo_server.py as a subprocess and gives its tools to the
agent. Swap the `MCPTools.stdio(...)` line for any other server, for example:

    MCPTools.stdio("npx", "-y", "@modelcontextprotocol/server-filesystem", ".")
    MCPTools("https://example.com/mcp")          # Streamable HTTP

pip install "agentloom[mcp,anthropic]"      # or [mcp,openai]
python examples/05_mcp.py
"""

import asyncio
import sys
from pathlib import Path

from _provider import provider_from_env

from agentloom import Agent, ConsoleTracer, Provider, ToolCall, ToolContext
from agentloom.mcp import MCPTools

SERVER = Path(__file__).with_name("mcp_demo_server.py")


def inventory_tools() -> MCPTools:
    # Reading stock is free; anything that changes it goes through the approver.
    return MCPTools.stdio(sys.executable, str(SERVER), requires_approval=["reserve"])


async def ask_user(call: ToolCall, ctx: ToolContext) -> bool:
    answer = await asyncio.to_thread(
        input, f"\n  Allow {call.name}({dict(call.arguments)})? [y/N] "
    )
    return answer.strip().lower() == "y"


def build_agent(provider: Provider, tools: MCPTools, approval=ask_user) -> Agent:
    return Agent(
        provider,
        instructions="You manage a small warehouse. Check stock before reserving anything.",
        tools=tools,
        approval=approval,
        hooks=[ConsoleTracer()],
    )


async def main() -> None:
    async with inventory_tools() as tools:
        agent = build_agent(provider_from_env(), tools)
        result = await agent.run("Reserve 5 widgets and 2 gadgets for order A-1001.")
        print("\n" + result.output)


if __name__ == "__main__":
    asyncio.run(main())
