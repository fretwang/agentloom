"""Test doubles. Use :class:`ScriptedProvider` to exercise agents without a model."""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from typing import Any

from .errors import ConfigurationError
from .providers.base import Completed, Provider, ProviderChunk, TextChunk
from .types import (
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    Usage,
)

Step = (
    str
    | ToolCall
    | Sequence[ToolCall]
    | ModelResponse
    | BaseException
    | Callable[[ModelRequest], Any]
)

_ids = itertools.count(1)


def call(name: str, **arguments: Any) -> ToolCall:
    """Build a tool call with a unique id: ``call("add", a=1, b=2)``."""
    return ToolCall(f"call_{next(_ids)}", name, arguments)


class ScriptedProvider(Provider):
    """Replays a fixed script, one step per model call.

    Each step is one of:

    * ``str``: a final text answer
    * ``ToolCall`` or a list of them: the model asks for tools
    * ``ModelResponse``: returned as is (to control stop reason or usage)
    * an exception instance: raised
    * a function ``request -> step``: computed from the request

    Every request is recorded in ``requests``. Running past the end of the
    script raises, which catches agents that loop longer than the test expects.
    """

    name = "scripted"

    def __init__(self, script: Iterable[Step], *, usage: Usage | None = None) -> None:
        self._script = list(script)
        self._usage = usage or Usage(input_tokens=10, output_tokens=5)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderChunk]:
        self.requests.append(request)
        if len(self.requests) > len(self._script):
            raise ConfigurationError(
                f"ScriptedProvider: model was called {len(self.requests)} times "
                f"but the script has {len(self._script)} steps"
            )
        step: Any = self._script[len(self.requests) - 1]
        if callable(step) and not isinstance(step, BaseException):
            step = step(request)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, ModelResponse):
            yield Completed(step)
            return
        if isinstance(step, str):
            midpoint = len(step) // 2
            for piece in (step[:midpoint], step[midpoint:]):
                if piece:
                    yield TextChunk(piece)
            message = Message("assistant", [TextBlock(step)])
            yield Completed(ModelResponse(message, StopReason.END_TURN, self._usage))
            return
        calls = [step] if isinstance(step, ToolCall) else list(step)
        yield Completed(
            ModelResponse(Message("assistant", list(calls)), StopReason.TOOL_USE, self._usage)
        )
