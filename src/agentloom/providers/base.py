"""The provider contract: the one seam between the agent loop and a model API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..errors import ProviderError
from ..types import ModelRequest, ModelResponse


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingChunk:
    text: str


@dataclass(frozen=True, slots=True)
class Completed:
    response: ModelResponse


ProviderChunk = TextChunk | ThinkingChunk | Completed


class Provider(ABC):
    """Adapter for one model API.

    Implement :meth:`stream`. It yields zero or more incremental chunks and
    then exactly one :class:`Completed` carrying the full response. A provider
    without streaming support yields only ``Completed``.

    Failures must be raised as :class:`~agentloom.errors.ProviderError` with
    ``retryable`` set, so the agent's retry policy can act on them. Providers
    should not retry internally; the agent owns retries so they are visible
    as events.
    """

    name: str = "provider"

    @abstractmethod
    def stream(self, request: ModelRequest) -> AsyncIterator[ProviderChunk]:
        """Send ``request`` and yield chunks, ending with ``Completed``."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        async for chunk in self.stream(request):
            if isinstance(chunk, Completed):
                return chunk.response
        raise ProviderError(f"{self.name}: stream ended without a final response", retryable=True)

    async def aclose(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release network resources. Override when the provider owns a client."""
