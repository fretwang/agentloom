"""Ready-made observers built on :class:`~agentloom.hooks.Hooks`."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from enum import Enum
from typing import Any, TextIO

from .events import (
    Event,
    ModelResponded,
    RetryScheduled,
    RunFinished,
    RunStarted,
    TextDelta,
    ThinkingDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from .hooks import Hooks


def event_to_dict(event: Event) -> dict[str, Any]:
    """A JSON-friendly rendering of an event, tagged with its type name."""
    return {"event": type(event).__name__, **dataclasses.asdict(event)}


def _json_default(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else str(value)


class JsonlTracer(Hooks):
    """Append every event except streaming deltas to a JSON Lines file.

    One line per event with a timestamp: enough to reconstruct what a run did,
    which tool calls it made with which arguments, and what each step cost.
    The file contains prompts and tool data; treat it as sensitive.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)

    async def on_event(self, event: Event) -> None:
        if isinstance(event, (TextDelta, ThinkingDelta)):
            return
        record = {"ts": time.time(), **event_to_dict(event)}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


class ConsoleTracer(Hooks):
    """Print a one-line-per-action trace, for development. ASCII only, so any console works."""

    def __init__(self, stream: TextIO | None = None, max_chars: int = 200) -> None:
        self.stream = stream or sys.stderr
        self.max_chars = max_chars

    def _short(self, text: str) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= self.max_chars else flat[: self.max_chars] + "..."

    async def on_event(self, event: Event) -> None:
        line: str | None = None
        if isinstance(event, RunStarted):
            line = f"> run {event.run_id[:8]} ({event.agent})"
        elif isinstance(event, ModelResponded):
            usage = event.response.usage
            line = (
                f"  step {event.step}: {event.response.stop_reason.value} "
                f"[in {usage.input_tokens} / out {usage.output_tokens} / "
                f"cached {usage.cache_read_tokens}]"
            )
        elif isinstance(event, ToolCallStarted):
            arguments = json.dumps(dict(event.call.arguments), ensure_ascii=False, default=str)
            line = f"  -> {event.call.name}({self._short(arguments)})"
        elif isinstance(event, ToolCallFinished):
            mark = "!!" if event.result.is_error else "ok"
            line = f"  {mark} {event.call.name} {event.duration:.2f}s: " + self._short(
                event.result.content
            )
        elif isinstance(event, RetryScheduled):
            line = f"  retry {event.attempt} in {event.delay:.1f}s: {self._short(event.error)}"
        elif isinstance(event, RunFinished):
            result = event.result
            line = (
                f"= {result.status.value} after {result.steps} step(s), "
                f"{result.usage.total_tokens} tokens"
            )
        if line is not None:
            print(line, file=self.stream)
