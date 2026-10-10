"""Context strategies decide which part of the history the model sees each step.

The stored history is never modified; a strategy returns a *view*. The default
keeps everything, which is the right choice until a conversation approaches
the model's context window: an append-only history keeps provider prompt
caches warm and keeps provider-bound reasoning blocks valid.

Strategies that drop messages do so in large, infrequent cuts rather than
sliding one message at a time, so the prefix stays byte-stable (and cacheable)
between cuts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import ConfigurationError
from .providers.base import Provider
from .types import Message, ModelRequest, TextBlock


@dataclass(frozen=True, slots=True)
class ContextView:
    messages: Sequence[Message]
    # True if anything was dropped or rewritten relative to the stored history.
    edited: bool = False


class ContextStrategy(ABC):
    @abstractmethod
    async def prepare(self, messages: Sequence[Message]) -> ContextView:
        """Return the messages to send for the next model call."""


class KeepAll(ContextStrategy):
    """Send the full history. The default."""

    async def prepare(self, messages: Sequence[Message]) -> ContextView:
        return ContextView(messages)


def _cut_index(messages: Sequence[Message], max_messages: int, keep: int) -> int:
    """Where the retained tail starts; 0 means keep everything.

    The cut advances in steps of ``max_messages - keep`` so it stays put
    between cuts, then moves forward to the next message that can legally
    start a conversation: a user message that is not a tool result.
    """
    count = len(messages)
    if count <= max_messages:
        return 0
    step = max_messages - keep
    cut = ((count - keep) // step) * step
    while cut < count:
        candidate = messages[cut]
        if candidate.role == "user" and not candidate.tool_results:
            return cut
        cut += 1
    return 0  # no safe boundary in range: keep everything rather than send an invalid history


def _check_window(max_messages: int, keep: int) -> None:
    if keep < 1 or keep >= max_messages:
        raise ConfigurationError("require 1 <= keep < max_messages")


class SlidingWindow(ContextStrategy):
    """Drop the oldest messages once the history exceeds ``max_messages``.

    When the limit is crossed the view is cut back to roughly ``keep``
    messages. Dropped content is gone from the model's view; use
    :class:`Summarizing` if earlier turns matter.
    """

    def __init__(self, max_messages: int = 80, keep: int = 40) -> None:
        _check_window(max_messages, keep)
        self.max_messages = max_messages
        self.keep = keep

    async def prepare(self, messages: Sequence[Message]) -> ContextView:
        cut = _cut_index(messages, self.max_messages, self.keep)
        if cut == 0:
            return ContextView(messages)
        return ContextView(messages[cut:], edited=True)


_SUMMARY_INSTRUCTIONS = (
    "You are compacting the earlier part of a conversation between a user and an AI agent "
    "so the agent can continue the work without it. Write a summary that preserves: the "
    "user's goals and constraints, decisions made and why, facts and identifiers the agent "
    "learned (names, paths, IDs, numbers), what has been completed, and what is still open. "
    "Be specific and complete rather than brief. Output only the summary."
)


class Summarizing(ContextStrategy):
    """Replace the oldest messages with a model-written summary.

    Cuts happen at the same points as :class:`SlidingWindow`; each cut costs one
    extra model call, whose result is cached for the life of this object.
    """

    def __init__(
        self,
        provider: Provider,
        max_messages: int = 80,
        keep: int = 40,
        instructions: str = _SUMMARY_INSTRUCTIONS,
    ) -> None:
        _check_window(max_messages, keep)
        self.provider = provider
        self.max_messages = max_messages
        self.keep = keep
        self.instructions = instructions
        self._summaries: dict[int, str] = {}

    async def prepare(self, messages: Sequence[Message]) -> ContextView:
        cut = _cut_index(messages, self.max_messages, self.keep)
        if cut == 0:
            return ContextView(messages)
        if cut not in self._summaries:
            self._summaries[cut] = await self._summarize(messages[:cut])
        note = TextBlock(
            "<summary_of_earlier_conversation>\n"
            f"{self._summaries[cut]}\n"
            "</summary_of_earlier_conversation>"
        )
        first = messages[cut]
        merged = Message("user", [note, *first.content])
        return ContextView([merged, *messages[cut + 1 :]], edited=True)

    async def _summarize(self, messages: Sequence[Message]) -> str:
        transcript = "\n\n".join(_render(m) for m in messages)
        request = ModelRequest(
            system=self.instructions,
            messages=[Message.user(f"<conversation>\n{transcript}\n</conversation>")],
        )
        response = await self.provider.complete(request)
        return response.message.text.strip()


def _render(message: Message) -> str:
    parts: list[str] = []
    if message.text:
        parts.append(message.text)
    for call in message.tool_calls:
        parts.append(f"[called tool {call.name} with {dict(call.arguments)!r}]")
    for result in message.tool_results:
        label = "tool error" if result.is_error else "tool result"
        parts.append(f"[{label}: {result.content}]")
    return f"{message.role.upper()}: " + "\n".join(parts)
