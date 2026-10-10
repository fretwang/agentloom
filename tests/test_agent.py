import asyncio

import pytest

from agentloom import (
    Agent,
    Approval,
    ConfigurationError,
    Hooks,
    InMemorySession,
    Message,
    ModelResponded,
    ModelResponse,
    ProviderError,
    RetryPolicy,
    RetryScheduled,
    RunFailed,
    RunFinished,
    RunLimits,
    RunStarted,
    RunStatus,
    StepStarted,
    StopReason,
    TextBlock,
    TextDelta,
    ToolCall,
    ToolCallFinished,
    ToolCallStarted,
    ToolContext,
    ToolDenied,
    ToolError,
    Usage,
    tool,
)
from agentloom.testing import ScriptedProvider, call


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def results_of(provider: ScriptedProvider, request_index: int = -1):
    return provider.requests[request_index].messages[-1].tool_results


async def test_plain_answer():
    provider = ScriptedProvider(["Hello!"])
    result = await Agent(provider, instructions="Be brief.").run("Hi")

    assert result.ok and result.status is RunStatus.COMPLETED
    assert result.output == "Hello!"
    assert result.steps == 1
    assert [m.role for m in result.messages] == ["user", "assistant"]
    assert provider.requests[0].system == "Be brief."
    assert result.raise_for_status() is result


async def test_tool_loop_feeds_results_back():
    provider = ScriptedProvider([call("add", a=2, b=3), "It is 5."])
    result = await Agent(provider, tools=[add]).run("2+3?")

    assert result.output == "It is 5."
    assert result.steps == 2
    assert result.usage == Usage(input_tokens=20, output_tokens=10)
    (tool_result,) = results_of(provider)
    assert tool_result.content == "5" and not tool_result.is_error
    assert [spec.name for spec in provider.requests[0].tools] == ["add"]


async def test_all_results_of_a_step_go_back_in_one_message():
    provider = ScriptedProvider([[call("add", a=1, b=1), call("add", a=2, b=2)], "done"])
    result = await Agent(provider, tools=[add]).run("go")

    assert [m.role for m in result.messages] == ["user", "assistant", "user", "assistant"]
    assert [r.content for r in results_of(provider)] == ["2", "4"]


async def test_parallel_tools_overlap_and_serial_tools_do_not():
    running = peak = 0

    async def body() -> str:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return "ok"

    @tool
    async def read() -> str:
        """Read-only."""
        return await body()

    @tool(parallel=False)
    async def write() -> str:
        """Has side effects."""
        return await body()

    await Agent(
        ScriptedProvider([[call("read"), call("read"), call("read")], "x"]), tools=[read, write]
    ).run("go")
    assert peak == 3

    peak = 0
    await Agent(
        ScriptedProvider([[call("read"), call("write"), call("read")], "x"]), tools=[read, write]
    ).run("go")
    assert peak == 1

    peak = 0
    limits = RunLimits(max_parallel_tools=2)
    await Agent(ScriptedProvider([[call("read")] * 4, "x"]), tools=[read], limits=limits).run("go")
    assert peak == 2


async def test_tool_failures_become_error_results_not_crashes():
    @tool
    def expected() -> str:
        """Fails in an expected way."""
        raise ToolError("date must be YYYY-MM-DD")

    @tool
    def bug() -> str:
        """Fails unexpectedly."""
        raise KeyError("missing")

    @tool(timeout=0.01)
    async def slow() -> str:
        """Too slow."""
        await asyncio.sleep(1)
        return "never"

    provider = ScriptedProvider(
        [
            [
                call("expected"),
                call("bug"),
                call("slow"),
                call("nope"),
                call("add", a="1", b=2),
            ],
            "recovered",
        ]
    )
    result = await Agent(provider, tools=[expected, bug, slow, add]).run("go")

    assert result.ok and result.output == "recovered"
    outcomes = results_of(provider)
    assert all(r.is_error for r in outcomes)
    assert outcomes[0].content == "date must be YYYY-MM-DD"
    assert outcomes[1].content == "KeyError: 'missing'"
    assert "Timed out after 0.01s" in outcomes[2].content
    assert "Unknown tool 'nope'" in outcomes[3].content and "add" in outcomes[3].content
    assert "$.a: expected integer, got str" in outcomes[4].content


async def test_long_tool_output_is_cut_with_a_notice():
    @tool
    def dump() -> str:
        """Return a lot."""
        return "x" * 500

    provider = ScriptedProvider([call("dump"), "ok"])
    await Agent(provider, tools=[dump], limits=RunLimits(max_tool_output_chars=100)).run("go")

    (outcome,) = results_of(provider)
    assert outcome.content.startswith("x" * 100 + "\n")
    assert "400 more characters" in outcome.content


async def test_step_limit_stops_a_looping_model():
    provider = ScriptedProvider([call("add", a=i, b=0) for i in range(10)])
    result = await Agent(provider, tools=[add], limits=RunLimits(max_steps=3)).run("go")

    assert result.status is RunStatus.MAX_STEPS and not result.ok
    assert result.steps == 3 and len(provider.requests) == 3
    assert result.messages[-1].tool_results  # history stays valid to continue from
    with pytest.raises(RunFailed, match="max_steps"):
        result.raise_for_status()


async def test_token_and_time_budgets():
    provider = ScriptedProvider([call("add", a=1, b=1)] * 5, usage=Usage(60, 40))
    result = await Agent(provider, tools=[add], limits=RunLimits(max_total_tokens=150)).run("go")
    assert result.status is RunStatus.TOKEN_BUDGET and result.steps == 2

    provider = ScriptedProvider([call("add", a=1, b=1)] * 5)
    result = await Agent(provider, tools=[add], limits=RunLimits(max_seconds=0)).run("go")
    assert result.status is RunStatus.TIMEOUT and result.steps == 0


async def test_identical_calls_are_refused_after_the_limit():
    executed = 0

    @tool
    def poll() -> str:
        """Check status."""
        nonlocal executed
        executed += 1
        return "pending"

    provider = ScriptedProvider([call("poll"), call("poll"), call("poll"), "gave up"])
    result = await Agent(provider, tools=[poll], limits=RunLimits(max_repeated_calls=2)).run("go")

    assert result.ok and executed == 2
    (third,) = results_of(provider)
    assert third.is_error and "already been made 2 times" in third.content


async def test_retry_then_success_and_exhaustion():
    waits: list[float] = []

    async def fake_sleep(delay: float) -> None:
        waits.append(delay)

    flaky = ProviderError("overloaded", retryable=True, retry_after=2)
    provider = ScriptedProvider([flaky, flaky, "ok"])
    agent = Agent(provider, retry=RetryPolicy(max_attempts=3))
    agent._sleep = fake_sleep
    events = [event async for event in agent.stream("go")]

    retries = [e for e in events if isinstance(e, RetryScheduled)]
    assert [r.attempt for r in retries] == [1, 2] and waits == [2, 2]
    assert events[-1].result.output == "ok"

    agent = Agent(ScriptedProvider([flaky, flaky]), retry=RetryPolicy(max_attempts=2))
    agent._sleep = fake_sleep
    with pytest.raises(ProviderError, match="overloaded"):
        await agent.run("go")

    fatal = ProviderError("bad request", retryable=False, status_code=400)
    with pytest.raises(ProviderError, match="bad request"):
        await Agent(ScriptedProvider([fatal, "unreached"])).run("go")


def test_retry_delay_is_bounded():
    policy = RetryPolicy(max_attempts=10, base_delay=1, max_delay=8)
    error = ProviderError("x", retryable=True)
    assert all(0 <= policy.delay(error, attempt) <= 8 for attempt in range(1, 10))
    assert policy.delay(ProviderError("x", retryable=True, retry_after=99), 1) == 8


def response(content, stop: StopReason) -> ModelResponse:
    return ModelResponse(Message("assistant", content), stop, Usage(1, 1))


async def test_stop_reasons_map_to_statuses():
    refused = await Agent(ScriptedProvider([response([], StopReason.REFUSAL)])).run("go")
    assert refused.status is RunStatus.REFUSED and refused.output == ""
    assert [m.role for m in refused.messages] == ["user"]  # empty reply is not stored

    cut = await Agent(
        ScriptedProvider([response([TextBlock("partial")], StopReason.MAX_TOKENS)])
    ).run("go")
    assert cut.status is RunStatus.TRUNCATED and cut.output == "partial"

    overflow = await Agent(ScriptedProvider([response([], StopReason.CONTEXT_OVERFLOW)])).run("go")
    assert overflow.status is RunStatus.CONTEXT_OVERFLOW

    paused = ScriptedProvider([response([TextBlock("part 1 ")], StopReason.PAUSE), "part 2"])
    result = await Agent(paused).run("go")
    assert result.ok and result.steps == 2
    assert paused.requests[1].messages[-1].role == "assistant"


async def test_truncated_tool_calls_are_not_executed():
    executed = False

    @tool
    def write(text: str) -> str:
        """Write text."""
        nonlocal executed
        executed = True
        return "written"

    cut = response([ToolCall("c1", "write", {"text": "half a sen"})], StopReason.MAX_TOKENS)
    provider = ScriptedProvider([cut, "ok"])
    result = await Agent(provider, tools=[write]).run("go")

    assert result.ok and not executed
    (outcome,) = results_of(provider)
    assert outcome.is_error and "output limit" in outcome.content


async def test_approval_gate():
    @tool(requires_approval=True)
    def delete(path: str) -> str:
        """Delete a file."""
        return f"deleted {path}"

    script = lambda: ScriptedProvider([call("delete", path="/tmp/x"), "done"])  # noqa: E731

    provider = script()
    await Agent(provider, tools=[delete]).run("go")
    assert "no approver is configured" in results_of(provider)[0].content

    asked: list[str] = []

    async def approve(tool_call: ToolCall, ctx: ToolContext) -> bool:
        asked.append(tool_call.arguments["path"])
        return True

    provider = script()
    await Agent(provider, tools=[delete], approval=approve).run("go")
    assert asked == ["/tmp/x"] and results_of(provider)[0].content == "deleted /tmp/x"

    async def deny(tool_call: ToolCall, ctx: ToolContext) -> Approval:
        return Approval(False, "outside the workspace")

    provider = script()
    await Agent(provider, tools=[delete], approval=deny).run("go")
    outcome = results_of(provider)[0]
    assert outcome.is_error and "not approved: outside the workspace" in outcome.content


async def test_hooks_observe_and_intercept():
    class Policy(Hooks):
        def __init__(self) -> None:
            self.events: list[str] = []

        async def on_event(self, event) -> None:
            self.events.append(type(event).__name__)

        async def before_model(self, request):
            from dataclasses import replace

            return replace(request, system=(request.system or "") + " [checked]")

        async def before_tool(self, tool_call, ctx) -> None:
            if tool_call.arguments["a"] > 100:
                raise ToolDenied("operand too large")

        async def after_tool(self, tool_call, result, ctx):
            from dataclasses import replace

            return replace(result, content=result.content.replace("5", "five"))

    class Broken(Hooks):
        async def on_event(self, event) -> None:
            raise RuntimeError("tracer down")

    policy = Policy()
    provider = ScriptedProvider([[call("add", a=2, b=3), call("add", a=500, b=1)], "ok"])
    result = await Agent(provider, instructions="Base.", tools=[add], hooks=[Broken(), policy]).run(
        "go"
    )

    assert result.ok  # a failing observer does not fail the run
    assert provider.requests[0].system == "Base. [checked]"
    first, second = results_of(provider)
    assert first.content == "five"
    assert second.is_error and "blocked by policy. operand too large" in second.content
    assert policy.events[0] == "RunStarted" and policy.events[-1] == "RunFinished"


async def test_stream_event_order():
    provider = ScriptedProvider([call("add", a=1, b=2), "three"])
    events = [event async for event in Agent(provider, tools=[add]).stream("go")]

    assert [type(e) for e in events] == [
        RunStarted,
        StepStarted,
        ModelResponded,
        ToolCallStarted,
        ToolCallFinished,
        StepStarted,
        TextDelta,
        TextDelta,
        ModelResponded,
        RunFinished,
    ]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "three"
    assert len({e.run_id for e in events}) == 1


async def test_session_carries_conversation_across_runs():
    session = InMemorySession()
    provider = ScriptedProvider(["Nice to meet you, Ada.", "Your name is Ada."])
    agent = Agent(provider)

    await agent.run("I'm Ada.", session=session)
    second = await agent.run("What's my name?", session=session)

    assert [m.text for m in provider.requests[1].messages] == [
        "I'm Ada.",
        "Nice to meet you, Ada.",
        "What's my name?",
    ]
    assert len(second.messages) == 4 and len(second.new_messages) == 2
    assert await session.load() == second.messages


async def test_interrupted_run_resumes_from_session():
    session = InMemorySession()
    crash = RuntimeError("process died")

    @tool
    def charge(amount: int) -> str:
        """Charge the customer."""
        raise crash

    class Die(Hooks):
        async def before_tool(self, tool_call, ctx) -> None:
            raise crash

    agent = Agent(ScriptedProvider([call("charge", amount=5)]), tools=[charge], hooks=[Die()])
    with pytest.raises(RuntimeError, match="process died"):
        await agent.run("charge 5", session=session)

    saved = await session.load()
    assert saved[-1].role == "assistant" and saved[-1].tool_calls  # unanswered call on disk

    provider = ScriptedProvider(["I could not confirm the charge."])
    result = await Agent(provider, tools=[charge]).run(session=session)

    assert result.ok
    (repaired,) = results_of(provider, 0)
    assert repaired.is_error and "interrupted" in repaired.content
    assert [m.role for m in result.new_messages] == ["user", "assistant"]


async def test_run_requires_something_to_do():
    agent = Agent(ScriptedProvider([]))
    with pytest.raises(ConfigurationError, match="nothing to run"):
        await agent.run()
    done = InMemorySession([Message.user("hi"), Message.assistant("hello")])
    with pytest.raises(ConfigurationError, match="no pending user turn"):
        await agent.run(session=done)


async def test_deps_state_and_dynamic_instructions():
    @tool
    def remember(key: str, ctx: ToolContext) -> str:
        """Count calls per key for this run."""
        ctx.state[key] = ctx.state.get(key, 0) + 1
        return f"{ctx.deps['tenant']}:{ctx.state[key]}"

    provider = ScriptedProvider([call("remember", key="k"), call("remember", key="k"), "ok"])
    agent = Agent(provider, tools=[remember], instructions=lambda deps: f"Tenant {deps['tenant']}")
    await agent.run("go", deps={"tenant": "acme"})

    assert provider.requests[0].system == "Tenant acme"
    assert results_of(provider)[0].content == "acme:2"


async def test_sub_agent_as_tool_reports_usage_and_failures():
    researcher = Agent(ScriptedProvider(["Paris."], usage=Usage(100, 50)), name="researcher")
    lead_provider = ScriptedProvider(
        [call("researcher", task="Capital of France?"), "Paris"], usage=Usage(10, 5)
    )
    lead = Agent(lead_provider, tools=[researcher.as_tool(description="Looks things up.")])
    result = await lead.run("go")

    assert results_of(lead_provider)[0].content == "Paris."
    assert result.usage == Usage(120, 60)
    assert researcher.provider.requests[0].messages[0].text == "Capital of France?"

    stuck = Agent(
        ScriptedProvider([call("add", a=1, b=1)] * 2),
        name="stuck",
        tools=[add],
        limits=RunLimits(max_steps=2),
    )
    lead_provider = ScriptedProvider([call("stuck", task="loop"), "ok"])
    await Agent(lead_provider, tools=[stuck.as_tool(description="Loops.")]).run("go")
    outcome = results_of(lead_provider)[0]
    assert outcome.is_error and "stopped early (max_steps)" in outcome.content


def test_run_sync():
    result = Agent(ScriptedProvider(["hi"])).run_sync("hello")
    assert result.output == "hi"


async def test_run_sync_inside_event_loop_is_a_clear_error():
    with pytest.raises(ConfigurationError, match=r"await agent.run"):
        Agent(ScriptedProvider(["hi"])).run_sync("hello")
