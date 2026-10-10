"""Hooks: observe a run and intercept it at two well-defined points.

Subclass :class:`Hooks` and override what you need. Hooks run in the order
they were given to the agent.

* ``on_event`` is observation only. An exception raised from it is logged and
  swallowed, so a broken tracer cannot take down a run.
* ``before_model``, ``before_tool`` and ``after_tool`` are control points.
  Exceptions raised from them propagate and fail the run, except
  ``ToolDenied`` from ``before_tool``, which blocks that one call.
"""

from __future__ import annotations

from .events import Event
from .tools import ToolContext
from .types import ModelRequest, ToolCall, ToolResult


class Hooks:
    async def on_event(self, event: Event) -> None:
        """Called for every event of the run."""

    async def before_model(self, request: ModelRequest) -> ModelRequest | None:
        """Inspect or replace the request about to be sent. Return None to keep it.

        Rewriting ``system``, ``tools`` or earlier ``messages`` between steps
        defeats prompt caching; if you must, also set ``history_edited=True``.
        """
        return None

    async def before_tool(self, call: ToolCall, ctx: ToolContext) -> None:
        """Called before a tool runs. Raise ``ToolDenied`` to block the call."""

    async def after_tool(
        self, call: ToolCall, result: ToolResult, ctx: ToolContext
    ) -> ToolResult | None:
        """Inspect or replace a tool result (e.g. redact secrets). Return None to keep it."""
        return None
