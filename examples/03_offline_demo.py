"""Runs with no API key: a scripted stand-in plays the model.

Shows the same machinery a real run uses (tool validation, error feedback,
parallel calls, a sub-agent, tracing), and is the pattern to copy for testing
your own agents.

    python examples/03_offline_demo.py
"""

from agentloom import Agent, ConsoleTracer, ToolError, tool
from agentloom.testing import ScriptedProvider, call

STOCK = {"widget": 12, "gadget": 0}
PRICES = {"widget": 4.5, "gadget": 19.0}


@tool
def stock_level(item: str) -> int:
    """Units of an item currently in stock."""
    if item not in STOCK:
        raise ToolError(f"Unknown item {item!r}. Known items: {', '.join(STOCK)}.")
    return STOCK[item]


@tool
def unit_price(item: str) -> float:
    """Price of one unit, in USD."""
    return PRICES[item]


pricing_agent = Agent(
    ScriptedProvider([call("unit_price", item="widget"), "A widget costs $4.50."]),
    name="pricing_specialist",
    tools=[unit_price],
)

model = ScriptedProvider(
    [
        # Step 1: the "model" makes two calls at once; one has a wrong argument type.
        [call("stock_level", item="widget"), call("stock_level", item=42)],
        # Step 2: it reads the validation error, corrects itself, and delegates pricing.
        [call("stock_level", item="gizmo"), call("pricing_specialist", task="Price of a widget?")],
        # Step 3: final answer.
        "We have 12 widgets at $4.50 each. I could not find an item called 'gizmo'.",
    ]
)

agent = Agent(
    model,
    tools=[stock_level, pricing_agent.as_tool(description="Answers questions about prices.")],
    hooks=[ConsoleTracer()],
)

if __name__ == "__main__":
    result = agent.run_sync("How many widgets and gizmos do we have, and what does a widget cost?")
    print("\n" + result.output)
    print(f"[{result.status.value}: {result.steps} steps, {result.usage.total_tokens} tokens]")
