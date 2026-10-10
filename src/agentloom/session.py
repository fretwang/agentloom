"""Sessions persist conversation history so runs can continue and recover."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from .errors import SessionError
from .types import Message, ToolResult

_FORMAT_VERSION = 1

INTERRUPTED_NOTICE = (
    "This tool call was interrupted before a result was recorded. It may or may not "
    "have taken effect; check the current state before repeating it."
)


class Session(ABC):
    """Storage for one conversation. Implement ``load`` and ``save``.

    ``save`` receives the full history each time and must replace what is
    stored atomically: the agent calls it after every model response and every
    batch of tool results, and a crash must never leave a half-written history.
    """

    @abstractmethod
    async def load(self) -> list[Message]: ...

    @abstractmethod
    async def save(self, messages: Sequence[Message]) -> None: ...


class InMemorySession(Session):
    """Keeps history for the lifetime of the object. Good for chat loops and tests."""

    def __init__(self, messages: Sequence[Message] = ()) -> None:
        self._messages = list(messages)

    async def load(self) -> list[Message]:
        return list(self._messages)

    async def save(self, messages: Sequence[Message]) -> None:
        self._messages = list(messages)


class FileSession(Session):
    """Stores history as one JSON file, replaced atomically on every save."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    async def load(self) -> list[Message]:
        return await asyncio.to_thread(self._read)

    async def save(self, messages: Sequence[Message]) -> None:
        payload = {"version": _FORMAT_VERSION, "messages": [m.to_dict() for m in messages]}
        await asyncio.to_thread(self._write, payload)

    def _read(self) -> list[Message]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != _FORMAT_VERSION:
                raise SessionError(
                    f"{self.path}: unsupported session format version {data.get('version')!r}"
                )
            return [Message.from_dict(m) for m in data["messages"]]
        except SessionError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SessionError(f"cannot load session {self.path}: {exc}") from exc

    def _write(self, payload: dict[str, object]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except (OSError, TypeError, ValueError) as exc:
            raise SessionError(f"cannot save session {self.path}: {exc}") from exc


def repair_history(messages: list[Message]) -> list[Message]:
    """Make a history that was cut off mid-step valid to send again.

    If the process died after the model requested tools but before their
    results were saved, the history ends with unanswered tool calls, which
    model APIs reject. Answer them with an error so the model can re-check and
    decide what to do.
    """
    if not messages:
        return messages
    last = messages[-1]
    if last.role == "assistant" and last.tool_calls:
        results = [
            ToolResult(call.id, INTERRUPTED_NOTICE, is_error=True) for call in last.tool_calls
        ]
        return [*messages, Message("user", list(results))]
    return messages
