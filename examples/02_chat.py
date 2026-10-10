"""An interactive chat: streaming output, a persistent session, and human approval.

The conversation is saved to .sessions/chat.json after every step, so you can
stop the script and pick up where you left off.

    python examples/02_chat.py
"""

import asyncio
from pathlib import Path

from agentloom import (
    Agent,
    FileSession,
    RunFinished,
    RunLimits,
    TextDelta,
    ToolCall,
    ToolCallStarted,
    ToolContext,
    ToolError,
    tool,
)
from agentloom.providers.anthropic import AnthropicProvider

NOTES = Path(".sessions/notes")


def _note_path(name: str) -> Path:
    # The name comes from the model: never let it choose a path outside NOTES.
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ToolError("Note names may only contain letters, digits, '-' and '_'.")
    return NOTES / f"{name}.txt"


@tool
def list_notes() -> list[str]:
    """List the names of all saved notes."""
    return sorted(p.stem for p in NOTES.glob("*.txt"))


@tool
def read_note(name: str) -> str:
    """Read a saved note by name."""
    path = _note_path(name)
    if not path.exists():
        raise ToolError(f"No note named {name!r}. Call list_notes to see what exists.")
    return path.read_text(encoding="utf-8")


@tool(requires_approval=True, parallel=False)
def save_note(name: str, text: str) -> str:
    """Create or overwrite a note. The user is asked to confirm each save."""
    NOTES.mkdir(parents=True, exist_ok=True)
    _note_path(name).write_text(text, encoding="utf-8")
    return f"Saved note {name!r}."


async def ask_user(call: ToolCall, ctx: ToolContext) -> bool:
    print(f"\n  The agent wants to run {call.name}({dict(call.arguments)})")
    answer = await asyncio.to_thread(input, "  Allow? [y/N] ")
    return answer.strip().lower() == "y"


async def main() -> None:
    agent = Agent(
        AnthropicProvider(),
        instructions="You help the user keep short notes. Check existing notes before writing.",
        tools=[list_notes, read_note, save_note],
        approval=ask_user,
        limits=RunLimits(max_steps=15),
    )
    session = FileSession(".sessions/chat.json")

    print("Chat with the notes agent. Empty line or Ctrl+C to quit.")
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nyou> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input.strip():
            break

        print("agent> ", end="", flush=True)
        async for event in agent.stream(user_input, session=session):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStarted):
                print(f"\n  [{event.call.name}]", end="", flush=True)
            elif isinstance(event, RunFinished) and not event.result.ok:
                print(f"\n  (stopped: {event.result.status.value})", end="")
        print()


if __name__ == "__main__":
    asyncio.run(main())
