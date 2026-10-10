"""Sessions, context strategies and tracing."""

import io
import json

import pytest

from agentloom import (
    Agent,
    ConfigurationError,
    ConsoleTracer,
    FileSession,
    JsonlTracer,
    Message,
    SessionError,
    SlidingWindow,
    Summarizing,
    TextBlock,
    ToolCall,
    ToolResult,
    tool,
)
from agentloom.session import repair_history
from agentloom.testing import ScriptedProvider, call


def conversation(turns: int) -> list[Message]:
    """user, assistant(tool call), user(tool result), assistant: four messages per turn."""
    messages: list[Message] = []
    for i in range(turns):
        messages += [
            Message.user(f"q{i}"),
            Message("assistant", [ToolCall(f"c{i}", "t", {"i": i})]),
            Message("user", [ToolResult(f"c{i}", f"r{i}")]),
            Message.assistant(f"a{i}"),
        ]
    return messages


async def test_file_session_round_trip(tmp_path):
    path = tmp_path / "nested" / "chat.json"
    session = FileSession(path)
    assert await session.load() == []

    history = [
        Message.user("你好"),
        Message(
            "assistant",
            [TextBlock("hi"), ToolCall("c1", "t", {"x": [1, 2]})],
            provider="anthropic",
            provider_data=[{"type": "thinking", "thinking": "", "signature": "sig"}],
        ),
        Message("user", [ToolResult("c1", "boom", is_error=True)]),
    ]
    await session.save(history)

    assert await FileSession(path).load() == history
    assert "你好" in path.read_text(encoding="utf-8")
    assert [p.name for p in path.parent.iterdir()] == ["chat.json"]  # no temp files left


async def test_file_session_reports_corruption(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionError, match="cannot load session"):
        await FileSession(path).load()

    path.write_text(json.dumps({"version": 99, "messages": []}), encoding="utf-8")
    with pytest.raises(SessionError, match="unsupported session format"):
        await FileSession(path).load()


def test_repair_history_only_touches_dangling_tool_calls():
    complete = conversation(1)
    assert repair_history(complete) is complete
    assert repair_history([]) == []

    dangling = complete[:2]
    repaired = repair_history(dangling)
    assert len(repaired) == 3
    (result,) = repaired[-1].tool_results
    assert result.tool_call_id == "c0" and result.is_error


async def test_sliding_window_cuts_in_stable_chunks_at_safe_boundaries():
    window = SlidingWindow(max_messages=8, keep=4)

    view = await window.prepare(conversation(2))  # 8 messages: at the limit
    assert not view.edited and len(view.messages) == 8

    starts = {}
    for size in range(9, 17):
        view = await window.prepare(conversation(4)[:size])
        assert view.edited
        first = view.messages[0]
        assert first.role == "user" and not first.tool_results
        starts[size] = first.text
    # The cut does not move with every new message, so the sent prefix stays cacheable.
    assert starts == {9: "q1", 10: "q1", 11: "q1", 12: "q2", 13: "q2", 14: "q2", 15: "q2", 16: "q3"}


async def test_window_never_sends_an_invalid_history():
    only_tool_traffic = [Message.user("start")] + conversation(3)[1:3] * 5
    view = await SlidingWindow(max_messages=4, keep=2).prepare(only_tool_traffic)
    assert not view.edited and len(view.messages) == len(only_tool_traffic)

    with pytest.raises(ConfigurationError):
        SlidingWindow(max_messages=4, keep=4)


async def test_summarizing_replaces_old_turns_and_caches_the_summary():
    summarizer = ScriptedProvider(["User asked q0 and q1; answers were a0 and a1."])
    strategy = Summarizing(summarizer, max_messages=8, keep=4)
    history = conversation(3)[:10]

    view = await strategy.prepare(history)
    again = await strategy.prepare([*history, Message("assistant", [TextBlock("more")])])

    assert len(summarizer.requests) == 1  # second call reused the cached summary
    assert view.edited and again.edited
    first = view.messages[0]
    assert first.role == "user"
    assert "<summary_of_earlier_conversation>" in first.text and first.text.endswith("q1")
    assert "a0 and a1" in first.text
    transcript = summarizer.requests[0].messages[0].text
    assert "USER: q0" in transcript and "[called tool t with {'i': 0}]" in transcript
    assert "q1" not in transcript  # the retained tail is not summarized


async def test_agent_marks_requests_built_from_an_edited_view():
    provider = ScriptedProvider(["ok"])
    agent = Agent(provider, context=SlidingWindow(max_messages=8, keep=4))
    from agentloom import InMemorySession

    await agent.run("latest", session=InMemorySession(conversation(3)))

    request = provider.requests[0]
    assert request.history_edited
    assert request.messages[0].text == "q2" and request.messages[-1].text == "latest"


async def test_tracers(tmp_path):
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    log = tmp_path / "trace.jsonl"
    console = io.StringIO()
    provider = ScriptedProvider([call("add", a=1, b=2), "three"])
    await Agent(provider, tools=[add], hooks=[JsonlTracer(log), ConsoleTracer(console)]).run("go")

    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    kinds = [r["event"] for r in records]
    assert kinds[0] == "RunStarted" and kinds[-1] == "RunFinished"
    assert "TextDelta" not in kinds
    finished = next(r for r in records if r["event"] == "ToolCallFinished")
    assert finished["call"]["arguments"] == {"a": 1, "b": 2}
    assert finished["result"]["content"] == "3"
    assert records[-1]["result"]["status"] == "completed"

    text = console.getvalue()
    assert '-> add({"a": 1, "b": 2})' in text and "ok add" in text and "= completed" in text
    assert text.isascii()
