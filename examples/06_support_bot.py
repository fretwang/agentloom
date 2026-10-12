"""Template: a customer-support agent you can adapt.

What it shows:

* the customer's identity comes from `deps`, never from the model
* refunds need approval: small ones are approved by policy, larger ones by a person
* each customer has a saved conversation that survives restarts
* every step is written to an audit log

Replace ORDERS/FAQ with your own data sources and run:

    python examples/06_support_bot.py
"""

import asyncio
from dataclasses import dataclass

from _provider import provider_from_env

from agentloom import (
    Agent,
    Approval,
    FileSession,
    JsonlTracer,
    Provider,
    RunFinished,
    RunLimits,
    TextDelta,
    ToolCall,
    ToolCallStarted,
    ToolContext,
    ToolError,
    tool,
)

AUTO_APPROVE_CENTS = 2_000

ORDERS = {
    "A-1001": {"customer": "c_42", "status": "delivered", "total_cents": 4_500, "refunded": 0},
    "A-1002": {"customer": "c_42", "status": "shipped", "total_cents": 12_900, "refunded": 0},
    "A-2001": {"customer": "c_77", "status": "processing", "total_cents": 900, "refunded": 0},
}
FAQ = {
    "returns": "Items can be returned within 30 days of delivery for a full refund.",
    "shipping": "Standard shipping takes 3-5 business days. Express takes 1-2.",
    "damaged": "Damaged items are refunded in full; no return is needed under $50.",
}


@dataclass
class Customer:
    id: str
    name: str


def _own_order(order_id: str, ctx: ToolContext) -> dict:
    order = ORDERS.get(order_id.upper())
    # Same message for "missing" and "someone else's": do not reveal which orders exist.
    if order is None or order["customer"] != ctx.deps.id:
        raise ToolError(f"No order {order_id!r} on this account.")
    return order


@tool
def list_orders(ctx: ToolContext) -> list[dict]:
    """List the signed-in customer's orders with their status."""
    return [
        {"order_id": order_id, "status": order["status"], "total_cents": order["total_cents"]}
        for order_id, order in ORDERS.items()
        if order["customer"] == ctx.deps.id
    ]


@tool
def order_status(order_id: str, ctx: ToolContext) -> dict:
    """Look up one of the customer's orders.

    Args:
        order_id: The order number, for example "A-1001".
    """
    order = _own_order(order_id, ctx)
    return {k: v for k, v in order.items() if k != "customer"}


@tool
def search_faq(topic: str) -> str:
    """Look up store policy. Topics: returns, shipping, damaged."""
    answer = FAQ.get(topic.lower())
    if answer is None:
        raise ToolError(f"No policy on {topic!r}. Topics: {', '.join(FAQ)}.")
    return answer


@tool(requires_approval=True, parallel=False)
def refund(order_id: str, amount_cents: int, reason: str, ctx: ToolContext) -> str:
    """Refund part or all of an order to the original payment method.

    Args:
        order_id: The order to refund.
        amount_cents: Amount in cents. Cannot exceed what is left of the order total.
        reason: Why the refund is being issued, in one sentence.
    """
    order = _own_order(order_id, ctx)
    remaining = order["total_cents"] - order["refunded"]
    if not 0 < amount_cents <= remaining:
        raise ToolError(f"Amount must be between 1 and {remaining} cents for this order.")
    order["refunded"] += amount_cents
    return f"Refunded ${amount_cents / 100:.2f} on {order_id.upper()}."


async def refund_policy(call: ToolCall, ctx: ToolContext) -> bool | Approval:
    amount = call.arguments["amount_cents"]
    if amount <= AUTO_APPROVE_CENTS:
        return True
    question = f"\n  Approve a ${amount / 100:.2f} refund on {call.arguments['order_id']}? [y/N] "
    answer = await asyncio.to_thread(input, question)
    return Approval(answer.strip().lower() == "y", reason="a supervisor declined the refund")


def build_agent(provider: Provider, approval=refund_policy, hooks=()) -> Agent:
    return Agent(
        provider,
        name="support",
        instructions=lambda customer: (
            f"You are a support agent for an online store, helping {customer.name}. "
            "Look things up before answering. Check the policy before offering a refund. "
            "If a refund is declined, tell the customer and offer to escalate; do not retry."
        ),
        tools=[list_orders, order_status, search_faq, refund],
        approval=approval,
        limits=RunLimits(max_steps=12),
        hooks=hooks,
    )


async def main() -> None:
    customer = Customer(id="c_42", name="Ada")
    agent = build_agent(provider_from_env(), hooks=[JsonlTracer(".sessions/support-audit.jsonl")])
    session = FileSession(f".sessions/support-{customer.id}.json")

    print(f"Support chat for {customer.name}. Empty line to quit.")
    while True:
        try:
            text = await asyncio.to_thread(input, "\nyou> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text.strip():
            break
        print("agent> ", end="", flush=True)
        async for event in agent.stream(text, session=session, deps=customer):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStarted):
                print(f"\n  [{event.call.name}]", end="", flush=True)
            elif isinstance(event, RunFinished) and not event.result.ok:
                print(f"\n  (stopped: {event.result.status.value})", end="")
        print()


if __name__ == "__main__":
    asyncio.run(main())
