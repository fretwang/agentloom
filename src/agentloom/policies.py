"""Retry and resource-limit policies. Plain data, so they are easy to tune and test."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .errors import ConfigurationError, ProviderError


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter for retryable provider errors.

    ``max_attempts`` counts the first try, so ``1`` disables retries. A
    provider-supplied ``retry_after`` takes precedence over the computed delay
    (capped at ``max_delay``).
    """

    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigurationError("RetryPolicy.max_attempts must be at least 1")

    def should_retry(self, error: ProviderError, attempt: int) -> bool:
        return error.retryable and attempt < self.max_attempts

    def delay(self, error: ProviderError, attempt: int) -> float:
        if error.retry_after is not None:
            return min(self.max_delay, max(0.0, error.retry_after))
        ceiling = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        return random.uniform(0, ceiling)


@dataclass(frozen=True, slots=True)
class RunLimits:
    """Hard ceilings that stop a run no matter what the model does.

    max_steps:           model calls per run.
    max_total_tokens:    cumulative tokens (input + output + cache) per run.
    max_seconds:         wall-clock budget, checked between steps.
    max_repeated_calls:  how many times the same tool call (name + arguments)
                         may execute in one run before further repeats are
                         answered with an error instead of being run.
    tool_timeout:        default per-call timeout in seconds; a tool's own
                         ``timeout`` overrides it.
    max_parallel_tools:  concurrency cap when a step requests several tools.
    max_tool_output_chars: longer tool outputs are cut, with a notice to the
                         model saying how much was removed.
    """

    max_steps: int = 25
    max_total_tokens: int | None = None
    max_seconds: float | None = None
    max_repeated_calls: int = 5
    tool_timeout: float | None = 120.0
    max_parallel_tools: int = 8
    max_tool_output_chars: int = 100_000

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigurationError("RunLimits.max_steps must be at least 1")
        if self.max_parallel_tools < 1:
            raise ConfigurationError("RunLimits.max_parallel_tools must be at least 1")
        if self.max_repeated_calls < 1:
            raise ConfigurationError("RunLimits.max_repeated_calls must be at least 1")
