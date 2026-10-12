"""The agent loop.

One :class:`Agent` is a reusable, stateless configuration: a provider,
instructions, tools and policies. Each call to ``run``/``stream`` is an
independent run; conversation state lives in the ``Session`` you pass in.

The loop, per step:

1. stop if a limit is hit
2. ask the context strategy for the view to send
3. call the model (streaming, with retries)
4. persist the reply; finish if it has no tool calls
5. validate, approve and execute the tool calls
6. persist the results and go to 1
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .context import ContextStrategy, KeepAll
from .errors import ConfigurationError, ProviderError, RunFailed, ToolDenied, ToolError
from .events import (
    Event,
    ModelResponded,
    RetryScheduled,
    RunFinished,
    RunStarted,
    StepStarted,
    TextDelta,
    ThinkingDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from .hooks import Hooks
from .output import ANSWER_NOT_ALONE, ANSWER_REMINDER, FINAL_ANSWER, OutputSpec
from .policies import RetryPolicy, RunLimits
from .providers.base import Provider, TextChunk, ThinkingChunk
from .session import Session, repair_history
from .tools import Tool, ToolContext, Toolset, render_output, validate_json
from .types import Message, ModelRequest, ModelResponse, StopReason, ToolCall, ToolResult, Usage

logger = logging.getLogger("agentloom")


class RunStatus(str, Enum):
    """How a run ended. Only ``COMPLETED`` means the model finished its answer."""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    TOKEN_BUDGET = "token_budget"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    TRUNCATED = "truncated"  # the final answer hit the output-token limit
    CONTEXT_OVERFLOW = "context_overflow"


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    status: RunStatus
    output: str
    messages: list[Message]  # full history, including earlier session turns
    new_messages: list[Message]  # what this run added
    usage: Usage
    steps: int
    data: Any = None  # the validated answer when the agent has an ``output_type``

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.COMPLETED

    def raise_for_status(self) -> RunResult:
        """Return self if the run completed, otherwise raise :class:`RunFailed`."""
        if not self.ok:
            raise RunFailed(f"run {self.run_id} ended with status {self.status.value!r}")
        return self


@dataclass(frozen=True, slots=True)
class Approval:
    approved: bool
    reason: str = ""


ApprovalHandler = Callable[[ToolCall, ToolContext], Awaitable["bool | Approval"]]
Instructions = str | Callable[[Any], str] | None


@dataclass(slots=True)
class _RunState:
    run_id: str
    deps: Any
    usage: Usage = field(default_factory=Usage)
    state: dict[str, Any] = field(default_factory=dict)
    call_counts: Counter[str] = field(default_factory=Counter)

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage


class Agent:
    """A model plus instructions, tools and policies.

    Args:
        provider: The model adapter.
        name: Used in events, logs and as the default sub-agent tool name.
        instructions: System prompt, or a function ``deps -> str`` evaluated
            once at the start of each run.
        tools: ``Tool`` objects or plain functions (wrapped with ``@tool``).
        limits: Hard ceilings for a run. See :class:`RunLimits`.
        retry: Backoff policy for retryable provider errors.
        context: Chooses what part of the history each model call sees.
        hooks: Observers and interceptors, called in order.
        approval: Awaited for every call to a tool with ``requires_approval``.
            Without one, such calls are refused.
        max_tokens: Output-token cap per model call; ``None`` uses the
            provider's default.
        output_type: Ask for a structured answer: a dataclass, pydantic model,
            any type a tool parameter may have, or a JSON Schema mapping. The
            validated answer is ``RunResult.data``.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        name: str = "agent",
        instructions: Instructions = None,
        tools: Iterable[Tool | Callable[..., Any]] = (),
        limits: RunLimits | None = None,
        retry: RetryPolicy | None = None,
        context: ContextStrategy | None = None,
        hooks: Sequence[Hooks] = (),
        approval: ApprovalHandler | None = None,
        max_tokens: int | None = None,
        output_type: Any = None,
    ) -> None:
        self.provider = provider
        self.name = name
        self.instructions = instructions
        self.tools = Toolset(tools)
        self.limits = limits or RunLimits()
        self.retry = retry or RetryPolicy()
        self.context = context or KeepAll()
        self.hooks = list(hooks)
        self.approval = approval
        self.max_tokens = max_tokens
        self._output = OutputSpec.of(output_type) if output_type is not None else None
        if self._output is not None and self.tools.get(FINAL_ANSWER) is not None:
            raise ConfigurationError(
                f"a tool named {FINAL_ANSWER!r} cannot be combined with output_type"
            )
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    # --- public API --------------------------------------------------------

    async def run(
        self,
        input: str | Message | None = None,
        *,
        session: Session | None = None,
        deps: Any = None,
    ) -> RunResult:
        """Run to completion and return the result.

        Pass ``input=None`` with a session to continue a run that was
        interrupted. ``deps`` is handed to tools via ``ToolContext.deps``.

        Limits, refusals and truncation are reported on ``result.status``.
        Raises :class:`ProviderError` if the model call fails beyond the retry
        policy; whatever was completed before that is already in the session.
        """
        result: RunResult | None = None
        async for event in self.stream(input, session=session, deps=deps):
            if isinstance(event, RunFinished):
                result = event.result
        assert result is not None  # stream always ends with RunFinished
        return result

    def run_sync(
        self,
        input: str | Message | None = None,
        *,
        session: Session | None = None,
        deps: Any = None,
    ) -> RunResult:
        """Blocking wrapper around :meth:`run` for scripts without an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(input, session=session, deps=deps))
        raise ConfigurationError(
            "run_sync() was called from a running event loop; use `await agent.run(...)`"
        )

    async def stream(
        self,
        input: str | Message | None = None,
        *,
        session: Session | None = None,
        deps: Any = None,
    ) -> AsyncIterator[Event]:
        """Run and yield events as they happen. The last event is ``RunFinished``."""
        async for event in self._iterate(input, session, deps):
            await self._notify(event)
            yield event

    def as_tool(self, *, name: str | None = None, description: str) -> Tool:
        """Expose this agent as a tool another agent can delegate to.

        The sub-agent starts with a fresh context and sees only the ``task``
        string, so the caller's instructions should tell it to write
        self-contained tasks. Its token usage is added to the caller's run.
        """

        async def delegate(arguments: Any, ctx: ToolContext) -> str:
            result = await self.run(arguments["task"], deps=ctx.deps)
            ctx.report_usage(result.usage)
            if not result.ok:
                partial = f" Partial output: {result.output}" if result.output else ""
                raise ToolError(
                    f"sub-agent {self.name!r} stopped early ({result.status.value}).{partial}"
                )
            return result.output

        return Tool(
            name=name or self.name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "A complete, self-contained description of the work. The "
                            "recipient cannot see this conversation."
                        ),
                    }
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            handler=delegate,
        )

    # --- the loop ----------------------------------------------------------

    async def _iterate(
        self, input: str | Message | None, session: Session | None, deps: Any
    ) -> AsyncIterator[Event]:
        stored = await session.load() if session is not None else []
        base = len(stored)
        history = list(repair_history(stored))
        if input is not None:
            history.append(Message.user(input) if isinstance(input, str) else input)
        if not history:
            raise ConfigurationError("nothing to run: pass an input or a session with history")
        if history[-1].role != "user":
            raise ConfigurationError("nothing to run: the session has no pending user turn")

        run = _RunState(run_id=uuid.uuid4().hex, deps=deps)
        system = self.instructions(deps) if callable(self.instructions) else self.instructions
        output = self._output
        specs = self.tools.specs() + ([output.spec] if output is not None else [])
        started = time.monotonic()
        step = 0
        status: RunStatus
        data: Any = None
        answer: str | None = None

        async def persist() -> None:
            if session is not None:
                await session.save(history)

        await persist()
        yield RunStarted(run.run_id, self.name)

        while True:
            limits = self.limits
            if step >= limits.max_steps:
                status = RunStatus.MAX_STEPS
                break
            if (
                limits.max_total_tokens is not None
                and run.usage.total_tokens >= limits.max_total_tokens
            ):
                status = RunStatus.TOKEN_BUDGET
                break
            if limits.max_seconds is not None and time.monotonic() - started >= limits.max_seconds:
                status = RunStatus.TIMEOUT
                break

            step += 1
            yield StepStarted(run.run_id, step)

            view = await self.context.prepare(history)
            request = ModelRequest(
                messages=tuple(view.messages),  # snapshot: history keeps growing
                system=system,
                tools=specs,
                max_tokens=self.max_tokens,
                history_edited=view.edited,
            )
            for hook in self.hooks:
                request = await hook.before_model(request) or request

            response: ModelResponse | None = None
            async for item in self._call_model(run.run_id, request):
                if isinstance(item, ModelResponse):
                    response = item
                else:
                    yield item
            assert response is not None

            run.add_usage(response.usage)
            if response.message.content or response.message.provider_data:
                history.append(response.message)
                await persist()
            yield ModelResponded(run.run_id, step, response)

            stop = response.stop_reason
            if stop is StopReason.REFUSAL:
                status = RunStatus.REFUSED
                break
            if stop is StopReason.CONTEXT_OVERFLOW:
                status = RunStatus.CONTEXT_OVERFLOW
                break
            if stop is StopReason.PAUSE:
                continue  # the provider wants the same history sent again

            calls = response.message.tool_calls
            if not calls:
                if stop is StopReason.MAX_TOKENS:
                    status = RunStatus.TRUNCATED
                    break
                if output is None:
                    status = RunStatus.COMPLETED
                    break
                # A structured answer has to arrive through the answer tool.
                history.append(Message.user(ANSWER_REMINDER))
                await persist()
                continue

            if stop is StopReason.MAX_TOKENS:
                # Arguments may be cut off mid-value; running them could act on partial data.
                results = [
                    ToolResult(
                        call.id,
                        "Not run: the response hit the output limit, so these arguments "
                        "may be incomplete. Call the tool again with a smaller payload.",
                        is_error=True,
                    )
                    for call in calls
                ]
            else:
                answers = [c for c in calls if output is not None and c.name == FINAL_ANSWER]
                others = [c for c in calls if c not in answers]
                by_id: dict[str, ToolResult] = {}
                if others:
                    async for outcome in self._execute_tools(run, step, others):
                        if isinstance(outcome, list):
                            by_id.update((r.tool_call_id, r) for r in outcome)
                        else:
                            yield outcome
                for call in answers:
                    assert output is not None
                    yield ToolCallStarted(run.run_id, step, call)
                    if others or len(answers) > 1:
                        # The answer may depend on results the model has not seen yet.
                        verdict = ToolResult(call.id, ANSWER_NOT_ALONE, is_error=True)
                    else:
                        data, text, problems = output.parse(call.arguments)
                        if problems:
                            listing = "\n".join(f"- {p}" for p in problems)
                            verdict = ToolResult(
                                call.id, f"Invalid answer:\n{listing}", is_error=True
                            )
                        else:
                            answer = text
                            verdict = ToolResult(call.id, "Answer accepted.")
                    by_id[call.id] = verdict
                    yield ToolCallFinished(run.run_id, step, call, verdict, 0.0)
                results = [by_id[c.id] for c in calls]
            history.append(Message("user", list(results)))
            await persist()
            if answer is not None:
                status = RunStatus.COMPLETED
                break

        new_messages = history[base:]
        last_reply = next((m for m in reversed(new_messages) if m.role == "assistant"), None)
        result = RunResult(
            run_id=run.run_id,
            status=status,
            output=answer if answer is not None else (last_reply.text if last_reply else ""),
            messages=list(history),
            new_messages=new_messages,
            usage=run.usage,
            steps=step,
            data=data,
        )
        yield RunFinished(run.run_id, result)

    async def _call_model(
        self, run_id: str, request: ModelRequest
    ) -> AsyncIterator[Event | ModelResponse]:
        attempt = 0
        while True:
            attempt += 1
            try:
                async for chunk in self.provider.stream(request):
                    if isinstance(chunk, TextChunk):
                        yield TextDelta(run_id, chunk.text)
                    elif isinstance(chunk, ThinkingChunk):
                        yield ThinkingDelta(run_id, chunk.text)
                    else:
                        yield chunk.response
                        return
                raise ProviderError(
                    f"{self.provider.name}: stream ended without a final response",
                    retryable=True,
                )
            except ProviderError as error:
                if not self.retry.should_retry(error, attempt):
                    raise
                delay = self.retry.delay(error, attempt)
                logger.warning(
                    "model call failed (attempt %d), retrying in %.1fs: %s", attempt, delay, error
                )
                yield RetryScheduled(run_id, attempt, delay, str(error))
                await self._sleep(delay)

    # --- tools -------------------------------------------------------------

    async def _execute_tools(
        self, run: _RunState, step: int, calls: Sequence[ToolCall]
    ) -> AsyncIterator[Event | list[ToolResult]]:
        # Decide repeats up front, in call order, so the outcome does not depend on scheduling.
        over_limit: set[str] = set()
        for call in calls:
            key = call.name + json.dumps(call.arguments, sort_keys=True, default=str)
            run.call_counts[key] += 1
            if run.call_counts[key] > self.limits.max_repeated_calls:
                over_limit.add(call.id)

        resolved = [self.tools.get(call.name) for call in calls]
        concurrent = len(calls) > 1 and all(
            t is not None and t.parallel and not t.requires_approval for t in resolved
        )
        results: list[ToolResult] = []

        if not concurrent:
            for call in calls:
                yield ToolCallStarted(run.run_id, step, call)
                result, duration = await self._run_tool(run, call, call.id in over_limit)
                results.append(result)
                yield ToolCallFinished(run.run_id, step, call, result, duration)
            yield results
            return

        for call in calls:
            yield ToolCallStarted(run.run_id, step, call)
        gate = asyncio.Semaphore(self.limits.max_parallel_tools)

        async def guarded(call: ToolCall) -> tuple[ToolResult, float]:
            async with gate:
                return await self._run_tool(run, call, call.id in over_limit)

        tasks = [asyncio.ensure_future(guarded(call)) for call in calls]
        try:
            outcomes = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for call, (result, duration) in zip(calls, outcomes, strict=True):
            results.append(result)
            yield ToolCallFinished(run.run_id, step, call, result, duration)
        yield results

    async def _run_tool(
        self, run: _RunState, call: ToolCall, over_limit: bool
    ) -> tuple[ToolResult, float]:
        started = time.monotonic()
        ctx = ToolContext(
            run_id=run.run_id,
            agent_name=self.name,
            call=call,
            deps=run.deps,
            state=run.state,
            _usage_sink=run.add_usage,
        )
        result = await self._invoke(call, ctx, over_limit)
        for hook in self.hooks:
            result = await hook.after_tool(call, result, ctx) or result
        return result, time.monotonic() - started

    async def _invoke(self, call: ToolCall, ctx: ToolContext, over_limit: bool) -> ToolResult:
        def failed(message: str) -> ToolResult:
            return ToolResult(call.id, message, is_error=True)

        tool = self.tools.get(call.name)
        if tool is None:
            available = ", ".join(self.tools.names()) or "(none)"
            return failed(f"Unknown tool {call.name!r}. Available tools: {available}")
        if over_limit:
            return failed(
                f"Not run: this exact call has already been made "
                f"{self.limits.max_repeated_calls} times in this run. Repeating it will not "
                "change the outcome; use the earlier results or take a different approach."
            )
        problems = validate_json(tool.input_schema, call.arguments)
        if problems:
            return failed("Invalid arguments:\n" + "\n".join(f"- {p}" for p in problems))

        if tool.requires_approval:
            if self.approval is None:
                return failed("Not run: this tool requires approval and no approver is configured.")
            decision = await self.approval(call, ctx)
            if isinstance(decision, bool):
                decision = Approval(decision)
            if not decision.approved:
                reason = f": {decision.reason}" if decision.reason else ""
                return failed(
                    f"Not run: the call was not approved{reason}. Do not retry it unless asked."
                )
        try:
            for hook in self.hooks:
                await hook.before_tool(call, ctx)
        except ToolDenied as denied:
            return failed(f"Not run: blocked by policy. {denied}".rstrip())

        timeout = tool.timeout if tool.timeout is not None else self.limits.tool_timeout
        try:
            output = await asyncio.wait_for(tool.handler(call.arguments, ctx), timeout)
        except ToolError as error:
            return failed(str(error) or "The tool reported a failure.")
        except asyncio.TimeoutError:
            return failed(
                f"Timed out after {timeout:g}s. The operation may still have taken effect."
            )
        except Exception as error:
            logger.exception("tool %r raised", call.name)
            return failed(f"{type(error).__name__}: {error}")
        return ToolResult(call.id, self._clip(render_output(output)))

    def _clip(self, text: str) -> str:
        limit = self.limits.max_tool_output_chars
        if len(text) <= limit:
            return text
        removed = len(text) - limit
        return (
            f"{text[:limit]}\n\n[Output cut: {removed} more characters were not shown. "
            "Ask for a narrower result if you need the rest.]"
        )

    async def _notify(self, event: Event) -> None:
        for hook in self.hooks:
            try:
                await hook.on_event(event)
            except Exception:
                logger.exception("hook %s.on_event raised; ignoring", type(hook).__name__)
