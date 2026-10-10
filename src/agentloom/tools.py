"""Tools: turning Python functions into things a model can call.

``@tool`` derives the JSON Schema from the function signature and the
description from its docstring, so the signature *is* the interface the model
sees. Arguments are validated against that schema before the function runs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import re
import types
import typing
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union, overload

from .errors import ConfigurationError, ToolError
from .types import ToolCall, ToolSpec, Usage

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_NONE_TYPE = type(None)


@dataclass(slots=True)
class ToolContext:
    """Per-call context injected into tools that declare a ``ToolContext`` parameter."""

    run_id: str
    agent_name: str
    call: ToolCall
    deps: Any = None
    # Scratch space shared by every tool call in the same run.
    state: dict[str, Any] = field(default_factory=dict)
    _usage_sink: Callable[[Usage], None] | None = None

    def report_usage(self, usage: Usage) -> None:
        """Add model usage incurred inside a tool (e.g. a sub-agent) to the run total."""
        if self._usage_sink is not None:
            self._usage_sink(usage)


ToolHandler = Callable[[Mapping[str, Any], ToolContext], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class Tool:
    """A callable capability.

    ``handler`` receives arguments that already passed schema validation.
    Set ``parallel=False`` for tools with side effects that must not interleave
    with other calls in the same step. ``requires_approval=True`` routes each
    call through the agent's approval handler first.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: ToolHandler
    requires_approval: bool = False
    parallel: bool = True
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ConfigurationError(
                f"invalid tool name {self.name!r}: use 1-64 letters, digits, '_' or '-'"
            )
        if not self.description.strip():
            raise ConfigurationError(
                f"tool {self.name!r} needs a description; the model chooses tools by it"
            )

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


class Toolset:
    """An ordered, name-unique collection of tools."""

    def __init__(self, tools: Iterable[Tool | Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools:
            self.add(item)

    def add(self, item: Tool | Callable[..., Any]) -> Tool:
        resolved = item if isinstance(item, Tool) else tool(item)
        if resolved.name in self._tools:
            raise ConfigurationError(f"duplicate tool name: {resolved.name!r}")
        self._tools[resolved.name] = resolved
        return resolved

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


# --- @tool -----------------------------------------------------------------


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...
@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    parallel: bool = True,
    timeout: float | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...
def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    parallel: bool = True,
    timeout: float | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a :class:`Tool`.

    Use bare (``@tool``) or with options (``@tool(requires_approval=True)``).
    Sync functions run in a worker thread so they cannot block the event loop.
    """

    def build(func: Callable[..., Any]) -> Tool:
        return _tool_from_function(
            func,
            name=name,
            description=description,
            requires_approval=requires_approval,
            parallel=parallel,
            timeout=timeout,
        )

    return build(fn) if fn is not None else build


def _tool_from_function(
    func: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    requires_approval: bool,
    parallel: bool,
    timeout: float | None,
) -> Tool:
    tool_name = name or func.__name__
    summary, param_docs = _parse_docstring(inspect.getdoc(func) or "")
    try:
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise ConfigurationError(f"tool {tool_name!r}: cannot resolve type hints ({exc})") from exc

    properties: dict[str, Any] = {}
    required: list[str] = []
    converters: dict[str, Callable[[Any], Any]] = {}
    context_param: str | None = None

    for param in inspect.signature(func).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ConfigurationError(
                f"tool {tool_name!r}: *args/**kwargs cannot be described to a model"
            )
        annotation = hints.get(param.name, Any)
        if annotation is ToolContext:
            context_param = param.name
            continue
        try:
            schema = dict(_schema_for(annotation))
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"tool {tool_name!r}, parameter {param.name!r}: {exc}"
            ) from exc
        if param.name in param_docs and "description" not in schema:
            schema["description"] = param_docs[param.name]
        properties[param.name] = schema
        if param.default is param.empty:
            required.append(param.name)
        converter = _converter_for(annotation)
        if converter is not None:
            converters[param.name] = converter

    input_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    is_async = inspect.iscoroutinefunction(func)

    async def handler(arguments: Mapping[str, Any], ctx: ToolContext) -> Any:
        kwargs = dict(arguments)
        for key, convert in converters.items():
            if key in kwargs:
                try:
                    kwargs[key] = convert(kwargs[key])
                except Exception as exc:
                    raise ToolError(f"invalid value for {key!r}: {exc}") from exc
        if context_param is not None:
            kwargs[context_param] = ctx
        if is_async:
            return await func(**kwargs)
        return await asyncio.to_thread(func, **kwargs)

    return Tool(
        name=tool_name,
        description=description or summary,
        input_schema=input_schema,
        handler=handler,
        requires_approval=requires_approval,
        parallel=parallel,
        timeout=timeout,
    )


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None
    current: str | None = None
    for raw in doc.splitlines():
        line = raw.strip()
        header = re.fullmatch(r"([A-Za-z ]+):", line)
        if header and not raw.startswith((" ", "\t")):
            section = header.group(1).lower()
            current = None
            continue
        if section is None:
            summary_lines.append(line)
        elif section in ("args", "arguments", "parameters"):
            match = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", line)
            if match:
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current and line:
                params[current] = f"{params[current]} {line}".strip()
    return " ".join(part for part in summary_lines if part).strip(), params


def _strip_annotated(annotation: Any) -> Any:
    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def _schema_for(annotation: Any) -> Mapping[str, Any]:
    """Map a Python type annotation to JSON Schema."""
    if annotation is Any or annotation is inspect.Parameter.empty:
        return {}
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Annotated:
        schema = dict(_schema_for(args[0]))
        note = next((m for m in args[1:] if isinstance(m, str)), None)
        if note:
            schema["description"] = note
        return schema
    if origin is Union or origin is types.UnionType:
        return {"anyOf": [_schema_for(a) for a in args]}
    if origin is Literal:
        return {"enum": list(args)}
    if annotation is _NONE_TYPE:
        return {"type": "null"}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"enum": [member.value for member in annotation]}

    container = origin or annotation
    if container in (list, set, frozenset, Sequence, typing.Sequence):
        return {"type": "array", "items": _schema_for(args[0])} if args else {"type": "array"}
    if container is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _schema_for(args[0])}
        return {"type": "array"}
    if container in (dict, Mapping, typing.Mapping):
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _schema_for(args[1])}
        return {"type": "object"}

    if hasattr(annotation, "model_json_schema"):  # pydantic v2 model
        return typing.cast("Mapping[str, Any]", annotation.model_json_schema())
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        hints = typing.get_type_hints(annotation, include_extras=True)
        fields = dataclasses.fields(annotation)
        return {
            "type": "object",
            "properties": {f.name: _schema_for(hints.get(f.name, Any)) for f in fields},
            "required": [
                f.name
                for f in fields
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            ],
            "additionalProperties": False,
        }
    raise ConfigurationError(
        f"unsupported type {annotation!r}; use JSON-compatible types, Enum, Literal, "
        "a dataclass, or a pydantic model"
    )


def _converter_for(annotation: Any) -> Callable[[Any], Any] | None:
    """Return a function that turns validated JSON into the annotated Python type."""
    annotation = _strip_annotated(annotation)
    origin = typing.get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        members = [a for a in typing.get_args(annotation) if a is not _NONE_TYPE]
        if len(members) == 1:
            inner = _converter_for(members[0])
            if inner is not None:
                return lambda value: None if value is None else inner(value)
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    if hasattr(annotation, "model_validate"):
        return typing.cast("Callable[[Any], Any]", annotation.model_validate)
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        cls = annotation
        return lambda value: cls(**value)
    return None


# --- validation ------------------------------------------------------------

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, Mapping),
}


def validate_json(schema: Mapping[str, Any], value: Any, path: str = "$") -> list[str]:
    """Check ``value`` against the JSON Schema subset this module generates.

    Returns human-readable problems (empty when valid). Schemas using ``$ref``
    (nested pydantic models) are not walked here; the pydantic converter
    validates those when the tool runs.
    """
    if "$ref" in schema:
        return []
    if "anyOf" in schema:
        if any(not validate_json(option, value, path) for option in schema["anyOf"]):
            return []
        return [f"{path}: does not match any allowed type"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: must be one of {schema['enum']!r}, got {value!r}"]

    expected = schema.get("type")
    if isinstance(expected, str):
        check = _TYPE_CHECKS.get(expected)
        if check is not None and not check(value):
            return [f"{path}: expected {expected}, got {type(value).__name__}"]

    problems: list[str] = []
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            problems += validate_json(schema["items"], item, f"{path}[{index}]")
    if isinstance(value, Mapping) and expected == "object":
        properties: Mapping[str, Any] = schema.get("properties", {})
        for key in schema.get("required", ()):
            if key not in value:
                problems.append(f"{path}: missing required field {key!r}")
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                problems += validate_json(properties[key], item, f"{path}.{key}")
            elif extra is False:
                problems.append(f"{path}: unexpected field {key!r}")
            elif isinstance(extra, Mapping):
                problems += validate_json(extra, item, f"{path}.{key}")
    return problems


def render_output(value: Any) -> str:
    """Serialise a tool's return value for the model."""
    if value is None:
        return "OK"
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
