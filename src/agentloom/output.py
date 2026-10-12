"""Structured final output: a typed, validated answer instead of free text.

The answer travels as the arguments of one extra tool, ``final_answer``, so it
works with every provider that supports tool calling and is validated by the
same code that validates tool arguments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ConfigurationError
from .tools import _converter_for, _schema_for, render_output, validate_json
from .types import ToolSpec

FINAL_ANSWER = "final_answer"

_DESCRIPTION = (
    "Give your final answer. Call this exactly once, on its own, when the task is "
    "complete. Its arguments are the answer; do not answer in plain text instead."
)
ANSWER_REMINDER = f"Give your final answer by calling the `{FINAL_ANSWER}` tool."
ANSWER_NOT_ALONE = (
    f"Not accepted: call `{FINAL_ANSWER}` once and on its own, after every other tool "
    "call has returned."
)


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """How to ask for, validate and convert an agent's structured answer."""

    spec: ToolSpec
    wrapped: bool  # the schema is not an object, so the answer sits under "value"
    convert: Callable[[Any], Any] | None

    @classmethod
    def of(cls, output_type: Any) -> OutputSpec:
        """Build from a type annotation or a JSON Schema mapping."""
        convert: Callable[[Any], Any] | None = None
        if isinstance(output_type, Mapping):
            schema = dict(output_type)
        else:
            try:
                schema = dict(_schema_for(output_type))
            except ConfigurationError as exc:
                raise ConfigurationError(f"output_type: {exc}") from exc
            convert = _converter_for(output_type)
        wrapped = schema.get("type") != "object"
        if wrapped:
            schema = {
                "type": "object",
                "properties": {"value": schema},
                "required": ["value"],
                "additionalProperties": False,
            }
        return cls(ToolSpec(FINAL_ANSWER, _DESCRIPTION, schema), wrapped, convert)

    def parse(self, arguments: Mapping[str, Any]) -> tuple[Any, str, list[str]]:
        """Return ``(data, text, problems)``; ``problems`` is empty when the answer is valid."""
        problems = validate_json(self.spec.input_schema, arguments)
        if problems:
            return None, "", problems
        raw = arguments["value"] if self.wrapped else dict(arguments)
        data = raw
        if self.convert is not None:
            try:
                data = self.convert(raw)
            except Exception as exc:
                return None, "", [str(exc)]
        return data, render_output(raw), []
