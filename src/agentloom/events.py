"""Events emitted during a run, in order. ``Agent.stream`` yields them; hooks observe them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import ModelResponse, ToolCall, ToolResult

if TYPE_CHECKING:
    from .agent import RunResult


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: str
    agent: str


@dataclass(frozen=True, slots=True)
class StepStarted:
    run_id: str
    step: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A fragment of the model's visible answer."""

    run_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """A fragment of the model's reasoning, when the provider exposes it."""

    run_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RetryScheduled:
    """A model call failed and will be retried.

    Text already streamed for the failed attempt is void; clear it in a UI.
    """

    run_id: str
    attempt: int
    delay: float
    error: str


@dataclass(frozen=True, slots=True)
class ModelResponded:
    run_id: str
    step: int
    response: ModelResponse


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    run_id: str
    step: int
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolCallFinished:
    run_id: str
    step: int
    call: ToolCall
    result: ToolResult
    duration: float


@dataclass(frozen=True, slots=True)
class RunFinished:
    run_id: str
    result: RunResult


Event = (
    RunStarted
    | StepStarted
    | TextDelta
    | ThinkingDelta
    | RetryScheduled
    | ModelResponded
    | ToolCallStarted
    | ToolCallFinished
    | RunFinished
)
