"""The smallest useful agent: one model, one tool.

pip install "agentloom[anthropic]"
export ANTHROPIC_API_KEY=...        # PowerShell: $env:ANTHROPIC_API_KEY = "..."
python examples/01_quickstart.py
"""

from agentloom import Agent, ConsoleTracer, ToolError, tool
from agentloom.providers.anthropic import AnthropicProvider

ORDERS = {
    "A-1001": {"status": "shipped", "items": ["keyboard", "wrist rest"], "eta": "2026-10-14"},
    "A-1002": {"status": "processing", "items": ["monitor arm"], "eta": None},
}


@tool
def order_status(order_id: str) -> dict:
    """Look up an order's status, items and estimated delivery date.

    Args:
        order_id: The order number, for example "A-1001".
    """
    order = ORDERS.get(order_id.upper())
    if order is None:
        raise ToolError(f"No order {order_id!r}. Order numbers look like 'A-1001'.")
    return order


agent = Agent(
    AnthropicProvider(),  # claude-opus-5-5; reads credentials from the environment
    instructions="You are a concise customer-support assistant for an online store.",
    tools=[order_status],
    hooks=[ConsoleTracer()],  # prints each step and tool call to stderr
)

if __name__ == "__main__":
    result = agent.run_sync("Where is my order a-1001, and has A-1002 shipped yet?")
    print(result.output)
    print()
    print(f"[{result.status.value}: {result.steps} steps, {result.usage.total_tokens} tokens]")
