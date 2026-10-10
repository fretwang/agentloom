import asyncio
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

from agentloom import ConfigurationError, ToolCall, ToolContext, ToolError, Toolset, tool
from agentloom.tools import render_output, validate_json


def ctx() -> ToolContext:
    return ToolContext(run_id="r", agent_name="a", call=ToolCall("c", "t", {}))


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Address(BaseModel):
    city: str
    zip: str


@dataclass
class Point:
    x: int
    y: int = 0


def test_schema_from_signature_and_docstring():
    @tool
    def search(
        query: str,
        limit: int = 10,
        mode: Literal["fast", "deep"] = "fast",
        tags: list[str] | None = None,
        ratio: Annotated[float, "Between 0 and 1."] = 0.5,
    ) -> str:
        """Search the knowledge base.

        Use for factual questions about the product.

        Args:
            query: What to look for.
            limit (int): Maximum number of hits,
                at most 50.
        """
        return query

    assert search.name == "search"
    assert search.description == (
        "Search the knowledge base. Use for factual questions about the product."
    )
    schema = search.input_schema
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert props["query"] == {"type": "string", "description": "What to look for."}
    assert props["limit"] == {
        "type": "integer",
        "description": "Maximum number of hits, at most 50.",
    }
    assert props["mode"] == {"enum": ["fast", "deep"]}
    assert props["tags"] == {
        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
    }
    assert props["ratio"] == {"type": "number", "description": "Between 0 and 1."}


async def test_arguments_are_converted_to_declared_types():
    seen = {}

    @tool
    def place(color: Color, address: Address, point: Point, backup: Color | None = None) -> str:
        """Place a marker."""
        seen.update(color=color, address=address, point=point, backup=backup)
        return "ok"

    args = {
        "color": "red",
        "address": {"city": "Oslo", "zip": "0150"},
        "point": {"x": 1},
        "backup": None,
    }
    assert validate_json(place.input_schema, args) == []
    assert await place.handler(args, ctx()) == "ok"
    assert seen == {
        "color": Color.RED,
        "address": Address(city="Oslo", zip="0150"),
        "point": Point(1, 0),
        "backup": None,
    }


async def test_bad_nested_model_becomes_tool_error():
    @tool
    def place(address: Address) -> str:
        """Place a marker."""
        return "ok"

    with pytest.raises(ToolError, match="address"):
        await place.handler({"address": {"city": "Oslo"}}, ctx())


async def test_context_is_injected_and_hidden_from_schema():
    @tool
    async def whoami(context: ToolContext, greeting: str = "hi") -> str:
        """Report the caller."""
        return f"{greeting} {context.agent_name} {context.deps}"

    assert "context" not in whoami.input_schema["properties"]
    context = ctx()
    context.deps = "db"
    assert await whoami.handler({}, context) == "hi a db"


async def test_sync_tools_run_off_the_event_loop():
    main_thread = threading.get_ident()

    @tool
    def blocking() -> int:
        """Return the worker thread id."""
        return threading.get_ident()

    assert await blocking.handler({}, ctx()) != main_thread
    assert threading.get_ident() == main_thread
    await asyncio.sleep(0)


def test_decorator_options_and_overrides():
    @tool(
        name="rm", description="Delete a file.", requires_approval=True, parallel=False, timeout=5
    )
    def delete(path: str) -> None:
        return None

    assert (delete.name, delete.description) == ("rm", "Delete a file.")
    assert delete.requires_approval and not delete.parallel and delete.timeout == 5


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: tool(lambda x: x), "invalid tool name"),
        (lambda: tool(_no_doc), "needs a description"),
        (lambda: tool(_var_args), "kwargs"),
        (lambda: tool(_bad_type), "unsupported type"),
    ],
)
def test_misconfigured_tools_fail_at_definition_time(factory, message):
    with pytest.raises(ConfigurationError, match=message):
        factory()


def _no_doc(x: int) -> int:
    return x


def _var_args(**kwargs: int) -> int:
    """Has kwargs."""
    return 0


def _bad_type(x: complex) -> int:
    """Has an unsupported type."""
    return 0


def test_toolset_rejects_duplicates_and_wraps_functions():
    def ping() -> str:
        """Ping."""
        return "pong"

    tools = Toolset([ping])
    assert tools.names() == ["ping"] and len(tools) == 1
    with pytest.raises(ConfigurationError, match="duplicate"):
        tools.add(ping)


@pytest.mark.parametrize(
    "value, problem",
    [
        ({"q": "x"}, None),
        ({"q": "x", "n": 3, "tags": ["a"]}, None),
        ({}, "missing required field 'q'"),
        ({"q": 1}, "$.q: expected string, got int"),
        ({"q": "x", "n": True}, "$.n: expected integer, got bool"),
        ({"q": "x", "n": 1.5}, "$.n: expected integer, got float"),
        ({"q": "x", "tags": ["a", 2]}, "$.tags[1]: expected string, got int"),
        ({"q": "x", "mode": "slow"}, "$.mode: must be one of"),
        ({"q": "x", "extra": 1}, "unexpected field 'extra'"),
        ("nope", "$: expected object, got str"),
    ],
)
def test_validate_json(value, problem):
    schema = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "n": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "mode": {"enum": ["fast", "deep"]},
        },
        "required": ["q"],
        "additionalProperties": False,
    }
    problems = validate_json(schema, value)
    if problem is None:
        assert problems == []
    else:
        assert any(problem in p for p in problems), problems


def test_render_output():
    assert render_output(None) == "OK"
    assert render_output("text") == "text"
    assert render_output({"名": 1}) == '{"名": 1}'
    assert render_output(Point(1, 2)) == '{"x": 1, "y": 2}'
    assert render_output(Address(city="Oslo", zip="1")) == '{"city": "Oslo", "zip": "1"}'
