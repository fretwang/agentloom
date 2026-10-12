"""The quickstart agent on any OpenAI-compatible endpoint.

pip install "agentloom[openai]"

    # DeepSeek
    export OPENAI_BASE_URL=https://api.deepseek.com OPENAI_API_KEY=... MODEL=deepseek-chat
    # Ollama, running locally
    export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama MODEL=llama3.3

    # OpenAI
    export OPENAI_API_KEY=... MODEL=<a model id>

    python examples/04_openai_compatible.py

The OpenAI SDK reads OPENAI_BASE_URL and OPENAI_API_KEY itself. The model must
support tool calling.
"""

import os

from agentloom import Agent, ConsoleTracer, ToolError, tool
from agentloom.providers.openai import OpenAIProvider

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
    OpenAIProvider(os.environ["MODEL"]),
    instructions="You are a concise customer-support assistant for an online store.",
    tools=[order_status],
    hooks=[ConsoleTracer()],
)

if __name__ == "__main__":
    result = agent.run_sync("Where is my order a-1001, and has A-1002 shipped yet?")
    print(result.output)
    print()
    print(f"[{result.status.value}: {result.steps} steps, {result.usage.total_tokens} tokens]")
