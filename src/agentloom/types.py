"""Provider-neutral data model shared by agents, tools, and providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model's request to run a tool."""

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of a tool call, sent back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False


Block = TextBlock | ToolCall | ToolResult


@dataclass(slots=True)
class Message:
    """One conversation turn.

    ``content`` is the neutral view every provider understands. ``provider_data``
    is the producing provider's native payload (for example reasoning blocks
    that must be echoed back unchanged); only the provider named in
    ``provider`` reads it. It must be JSON-serialisable so sessions can persist.
    """

    role: Role
    content: list[Block]
    provider: str | None = None
    provider_data: Any = None

    @classmethod
    def user(cls, text: str) -> Message:
        return cls("user", [TextBlock(text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls("assistant", [TextBlock(text)])

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]

    @property
    def tool_results(self) -> list[ToolResult]:
        return [b for b in self.content if isinstance(b, ToolResult)]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "content": [_block_to_dict(b) for b in self.content],
        }
        if self.provider is not None:
            data["provider"] = self.provider
            data["provider_data"] = self.provider_data
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message:
        return cls(
            role=data["role"],
            content=[_block_from_dict(b) for b in data["content"]],
            provider=data.get("provider"),
            provider_data=data.get("provider_data"),
        )


def _block_to_dict(block: Block) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolCall):
        return {
            "type": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments": dict(block.arguments),
        }
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_call_id": block.tool_call_id,
        "content": block.content,
    }
    if block.is_error:
        result["is_error"] = True
    return result


def _block_from_dict(data: Mapping[str, Any]) -> Block:
    kind = data["type"]
    if kind == "text":
        return TextBlock(data["text"])
    if kind == "tool_call":
        return ToolCall(data["id"], data["name"], dict(data["arguments"]))
    if kind == "tool_result":
        return ToolResult(data["tool_call_id"], data["content"], bool(data.get("is_error", False)))
    raise ValueError(f"unknown block type: {kind!r}")


class StopReason(str, Enum):
    """Why the model stopped, normalised across providers."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    PAUSE = "pause"  # provider paused a long turn; send the history again to continue
    REFUSAL = "refusal"
    CONTEXT_OVERFLOW = "context_overflow"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What the model sees of a tool."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: Sequence[Message]
    system: str | None = None
    tools: Sequence[ToolSpec] = ()
    max_tokens: int | None = None
    # True when a context strategy dropped or rewrote earlier messages. Providers
    # whose reasoning blocks are bound to an unedited history must not replay them.
    history_edited: bool = False


@dataclass(frozen=True, slots=True)
class ModelResponse:
    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    request_id: str | None = None
