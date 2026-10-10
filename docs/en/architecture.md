# Architecture and extension

[简体中文](../zh/architecture.md)

Contents: [Design principles](#design-principles) · [Module map](#module-map) · [Life of a run](#life-of-a-run) · [Guarantees](#guarantees) · [Writing a provider](#writing-a-provider) · [Writing a session store](#writing-a-session-store) · [Writing a context strategy](#writing-a-context-strategy) · [What is not here](#what-is-not-here)

---

## Design principles

**1. One loop, visible.** The agent loop is a single function, `Agent._iterate`, of about 120 lines. There is no graph engine, no planner abstraction, and no hidden prompt: the only text the model receives is your instructions, your tool descriptions, and the conversation.

**2. Four seams, each a small interface.** Everything that varies between deployments sits behind one of four interfaces, each with one or two methods:

| Seam | Interface | Methods | Shipped implementations |
|---|---|---|---|
| Model API | `Provider` | `stream` | `AnthropicProvider`, `ScriptedProvider` |
| Persistence | `Session` | `load`, `save` | `InMemorySession`, `FileSession` |
| Context selection | `ContextStrategy` | `prepare` | `KeepAll`, `SlidingWindow`, `Summarizing` |
| Policy and observation | `Hooks` | `on_event`, `before_model`, `before_tool`, `after_tool` | `ConsoleTracer`, `JsonlTracer` |

The core imports no third-party package. The Anthropic SDK is imported only inside `AnthropicProvider`.

**3. Failures the model can fix go to the model; everything else is explicit.** A bad argument, an unknown tool, a tool exception, a timeout, a denied approval: each becomes an error result in the conversation, because the model can often recover. Conditions the caller must handle (limits, refusals, truncation) are a `RunStatus`. Only infrastructure failures and programming errors raise.

**4. Safe defaults.** A missing approver refuses. Truncated tool arguments are not executed. Every run has a step limit. Tool arguments are always validated. You opt out of safety; you do not opt in.

**5. Append-only history.** The stored conversation only grows. Context strategies produce views; they never rewrite what is stored. This keeps provider prompt caches effective, keeps provider-signed reasoning blocks valid, and makes the session a faithful audit record.

---

## Module map

```
src/agentloom/
├── types.py        Message, TextBlock, ToolCall, ToolResult, Usage, ModelRequest, ModelResponse
├── errors.py       Exception hierarchy
├── tools.py        @tool, Tool, Toolset, ToolContext, schema generation, validation
├── policies.py     RetryPolicy, RunLimits
├── session.py      Session, InMemorySession, FileSession, repair_history
├── context.py      ContextStrategy, KeepAll, SlidingWindow, Summarizing
├── events.py       Event dataclasses
├── hooks.py        Hooks base class
├── agent.py        Agent, RunResult, RunStatus, Approval: the loop
├── tracing.py      ConsoleTracer, JsonlTracer
├── testing.py      ScriptedProvider, call
└── providers/
    ├── base.py       Provider, TextChunk, ThinkingChunk, Completed
    └── anthropic.py  AnthropicProvider
```

Dependencies point one way: `agent.py` depends on everything else; nothing depends on `agent.py` except `events.py`, for one type annotation.

### The data model

`Message` has a `role` (`"user"` or `"assistant"`) and a list of blocks:

- `TextBlock(text)`
- `ToolCall(id, name, arguments)`, in assistant messages
- `ToolResult(tool_call_id, content, is_error)`, in user messages

This neutral view is what tools, hooks, sessions and context strategies work with. A message can also carry `provider_data`: the producing provider's native payload, which only that provider reads. This is how Claude's reasoning blocks are echoed back unchanged without any other part of the framework knowing they exist. `provider_data` must be JSON-serialisable so sessions can store it.

---

## Life of a run

```
agent.run(input, session, deps)
│
├─ load history from the session; repair a dangling tool call if the last run was cut off
├─ append the user message; save
│
└─ loop
   ├─ limit reached? ─────────────────────► status MAX_STEPS / TOKEN_BUDGET / TIMEOUT
   ├─ context.prepare(history) ─► view
   ├─ hooks.before_model(request)
   ├─ provider.stream(request)            (retryable ProviderError: back off, retry)
   │     └─ emits TextDelta / ThinkingDelta
   ├─ append the assistant message; save
   ├─ stop reason:
   │     refusal ─────────────────────────► status REFUSED
   │     context overflow ────────────────► status CONTEXT_OVERFLOW
   │     pause ───────────────────────────► next iteration (send the history again)
   │     no tool calls ───────────────────► status COMPLETED (or TRUNCATED)
   ├─ for each tool call (in parallel when every tool allows it):
   │     unknown tool?            ─► error result
   │     repeated too often?      ─► error result
   │     arguments invalid?       ─► error result
   │     requires approval?       ─► approver; denied ─► error result
   │     hooks.before_tool        ─► ToolDenied ─► error result
   │     run handler with timeout ─► result, or error result
   │     hooks.after_tool
   └─ append ONE user message holding all results; save
```

Two details matter for correctness:

- **All results of a step go back in a single message, in call order.** Splitting them across messages is accepted by model APIs but teaches models to stop making parallel calls.
- **When a response is cut off by the output limit while requesting tools, the calls are not executed.** A partly generated argument can parse as valid JSON with a truncated value. Each call is answered with an error asking the model to retry with a smaller payload.

---

## Guarantees

What the framework ensures, and where the guarantee ends.

| Guarantee | Mechanism | Boundary |
|---|---|---|
| A run terminates | `max_steps` is always set; optional token and time budgets | A single model call or tool call is bounded by its own timeout, not by the run budget |
| Tool code never sees arguments that violate its schema | `validate_json` before every call; pydantic and dataclass conversion | Semantic checks (does this ID belong to this user?) are your tool's job |
| A tool failure does not crash the run | Every exception from a handler becomes an error result | `BaseException` (cancellation, `KeyboardInterrupt`) propagates |
| The stored history can always be sent to the model again | Saved after each half-step; dangling tool calls repaired on load | Assumes one writer per session |
| A saved session file is never half-written | Temp file, `fsync`, atomic rename | `FileSession` only; other stores must provide their own atomicity |
| A crashed tool call is not silently re-run | Interrupted calls get an error result, and the model decides | The framework cannot know whether the side effect happened: make tools idempotent |
| Approval-gated tools do not run without approval | Checked in the loop, before `before_tool` hooks | Your approver's own logic |
| A failing observer does not fail the run | `on_event` exceptions are logged and swallowed | `before_model`, `before_tool`, `after_tool` are control points; their exceptions propagate |
| Retries are bounded and visible | `RetryPolicy.max_attempts`; `RetryScheduled` events | Text streamed during a failed attempt has already reached your UI; clear it on `RetryScheduled` |

### What is not guaranteed

- **Exactly-once tool execution.** Not achievable in general. The framework gives at-most-once per recorded call plus an honest "this may have happened" signal after a crash.
- **Protection from prompt injection.** Tool results are data, but a model can still be misled by text in them. Limit what tools can do, take identity from `deps`, and require approval for consequential actions. Do not rely on instructions for security.
- **Concurrency control on sessions.** Two runs on one session at once will overwrite each other. Serialise per conversation in your application.

---

## Writing a provider

A provider adapts one model API. Implement `stream`: yield optional `TextChunk`/`ThinkingChunk` items, then exactly one `Completed` with the full response. Raise `ProviderError` on failure, with `retryable` set correctly, and do not retry internally.

The skeleton below targets a hypothetical JSON API and shows every responsibility:

```python
from collections.abc import AsyncIterator

from agentloom import (
    Completed, Message, ModelRequest, ModelResponse, Provider, ProviderChunk, ProviderError,
    StopReason, TextBlock, TextChunk, ToolCall, ToolResult, Usage,
)


class AcmeProvider(Provider):
    name = "acme"   # stored on messages; identifies who may read provider_data

    def __init__(self, client, model: str) -> None:
        self.client, self.model = client, model

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderChunk]:
        payload = {
            "model": self.model,
            "system": request.system,
            "max_tokens": request.max_tokens or 4096,
            "messages": [self._encode(m) for m in request.messages],
            "tools": [
                {"name": t.name, "description": t.description, "parameters": dict(t.input_schema)}
                for t in request.tools
            ],
        }
        try:
            reply = await self.client.chat(payload)            # 1. call the API
        except self.client.RateLimited as exc:                 # 2. classify failures
            raise ProviderError(str(exc), retryable=True, status_code=429,
                                retry_after=exc.retry_after) from exc
        except self.client.ServerError as exc:
            raise ProviderError(str(exc), retryable=True, status_code=exc.status) from exc
        except self.client.ClientError as exc:
            raise ProviderError(str(exc), retryable=False, status_code=exc.status) from exc

        blocks: list[TextBlock | ToolCall | ToolResult] = []   # 3. decode into neutral blocks
        if reply["text"]:
            yield TextChunk(reply["text"])                     #    (a streaming API yields as it goes)
            blocks.append(TextBlock(reply["text"]))
        for item in reply.get("tool_calls", []):
            blocks.append(ToolCall(item["id"], item["name"], item["arguments"]))

        stop = {                                               # 4. normalise the stop reason
            "stop": StopReason.END_TURN,
            "tool_calls": StopReason.TOOL_USE,
            "length": StopReason.MAX_TOKENS,
        }.get(reply["finish_reason"], StopReason.OTHER)

        yield Completed(ModelResponse(                         # 5. finish with exactly one Completed
            message=Message("assistant", blocks, provider=self.name),
            stop_reason=stop,
            usage=Usage(reply["usage"]["in"], reply["usage"]["out"]),
            model=self.model,
        ))

    def _encode(self, message: Message) -> dict:
        ...  # map TextBlock / ToolCall / ToolResult to the API's message format
```

Checklist:

1. **Stop reasons.** Map "the output limit was hit" to `MAX_TOKENS` and safety refusals to `REFUSAL`. The loop's handling of truncated tool calls depends on `MAX_TOKENS` being reported.
2. **Tool-call arguments must be a dict.** If the API returns arguments as a JSON string, parse it; on a parse failure raise `ProviderError(retryable=True)` so the turn is re-issued.
3. **Tool results.** Some APIs want each result as its own message with a `tool` role. Expand `ToolResult` blocks accordingly in `_encode`; the framework always hands you all results of a step in one user message.
4. **`provider_data`.** If the API returns anything that must be sent back verbatim, store it in `Message.provider_data` (JSON-serialisable) and set `provider=self.name`. When `request.history_edited` is true, leave out anything that is only valid for an unedited history.
5. **Retries.** Turn off the SDK's own retries, or you will retry twice.
6. **Test it** the way `tests/test_anthropic_provider.py` does: run the real SDK against a fake HTTP transport and assert on the request body and the decoded response.

---

## Writing a session store

Implement `load` and `save`. `save` receives the entire history each time and must replace the stored value atomically.

```python
import json
from collections.abc import Sequence

from agentloom import Message, Session


class RedisSession(Session):
    def __init__(self, redis, conversation_id: str, ttl_seconds: int = 30 * 86400) -> None:
        self.redis, self.key, self.ttl = redis, f"agentloom:session:{conversation_id}", ttl_seconds

    async def load(self) -> list[Message]:
        raw = await self.redis.get(self.key)
        return [Message.from_dict(m) for m in json.loads(raw)] if raw else []

    async def save(self, messages: Sequence[Message]) -> None:
        payload = json.dumps([m.to_dict() for m in messages], ensure_ascii=False)
        await self.redis.set(self.key, payload, ex=self.ttl)   # a single SET is atomic
```

Notes:

- `Message.to_dict()` / `Message.from_dict()` are the stable serialisation; store their output as is.
- `save` is called twice per step with the full history. For very long conversations in a SQL store, an append-only table keyed by `(conversation_id, position)` avoids rewriting everything: insert only the rows past the stored count.
- To prevent concurrent runs on one conversation, take a lock in your application around `agent.run`, or add optimistic versioning in `save` and raise `SessionError` on conflict.
- Sessions hold user content. Apply the same access control, encryption and retention rules as for the rest of your user data.

---

## Writing a context strategy

Implement `prepare`. It receives the full stored history and returns a `ContextView`.

```python
from collections.abc import Sequence

from agentloom import ContextStrategy, ContextView, Message, ToolResult


class ElideOldToolOutput(ContextStrategy):
    """Replace the content of old, large tool results with a placeholder."""

    def __init__(self, keep_recent: int = 20, max_chars: int = 2_000) -> None:
        self.keep_recent, self.max_chars = keep_recent, max_chars

    async def prepare(self, messages: Sequence[Message]) -> ContextView:
        boundary = len(messages) - self.keep_recent
        edited = False
        view: list[Message] = []
        for index, message in enumerate(messages):
            if index < boundary and any(
                isinstance(b, ToolResult) and len(b.content) > self.max_chars
                for b in message.content
            ):
                edited = True
                message = Message(message.role, [
                    ToolResult(b.tool_call_id, "[old output removed]", b.is_error)
                    if isinstance(b, ToolResult) and len(b.content) > self.max_chars else b
                    for b in message.content
                ])
            view.append(message)
        return ContextView(view, edited=edited)
```

Rules a strategy must follow:

1. **Return `edited=True` whenever the view differs from the input.** Providers rely on it to drop content that is only valid for an unedited history.
2. **Keep the view valid.** It must start with a user message that is not a tool result, and every `ToolCall` must keep its `ToolResult`.
3. **Do not mutate the input** or the messages in it; build new `Message` objects.
4. **Be stable.** The same history should give the same view, and the view's prefix should change rarely. A strategy that changes the prefix on every call (like the example above, whose boundary moves each turn) defeats prompt caching; move the boundary in large steps, as `SlidingWindow` does.

---

## What is not here

Left out on purpose, to keep the core small and correct. Each can be built on the existing seams.

| Not included | How to add it |
|---|---|
| Providers for other model APIs | A `Provider` subclass, about 60 lines; see above |
| Structured final output (a typed, schema-validated answer) | A provider option for the API's structured-output feature, or a final "submit_answer" tool whose arguments are the answer |
| MCP client | A function that lists an MCP server's tools and wraps each one as a `Tool` whose handler forwards the call |
| Built-in tools (files, shell, web) | Ordinary `@tool` functions. They are omitted because safe versions depend on your sandbox and permissions |
| Durable execution across processes for long approvals | Sessions plus `agent.run(session=...)` to resume cover crash recovery; a workflow engine is the right tool for multi-day orchestration |
| Images and files in messages | A new block type in `types.py` plus provider encoding |
| Evaluation harness | `ScriptedProvider` covers deterministic tests; quality evaluation against real models needs a dataset and graders of your own |
| Token counting before a call | Budgets use the usage the provider reports after each call |
