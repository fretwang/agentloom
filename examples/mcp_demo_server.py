"""A tiny MCP server for examples/05_mcp.py. It speaks MCP over stdin/stdout."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("inventory")

STOCK = {"widget": 12, "gadget": 0, "sprocket": 40}


@server.tool()
def list_items() -> list[str]:
    """List every item the warehouse knows about."""
    return sorted(STOCK)


@server.tool()
def stock_level(item: str) -> int:
    """Units of an item currently in stock."""
    if item not in STOCK:
        raise ValueError(f"Unknown item {item!r}. Call list_items to see what exists.")
    return STOCK[item]


@server.tool()
def reserve(item: str, quantity: int) -> str:
    """Reserve units of an item for an order. This changes the stock level."""
    if STOCK.get(item, 0) < quantity:
        raise ValueError(f"Only {STOCK.get(item, 0)} units of {item!r} are in stock.")
    STOCK[item] -= quantity
    return f"Reserved {quantity} x {item}. {STOCK[item]} left."


if __name__ == "__main__":
    server.run("stdio")
