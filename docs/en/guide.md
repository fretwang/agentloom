# agentloom user guide

[简体中文](../zh/guide.md)

Contents: [First agent](#1-first-agent) · [Tools](#2-tools) · [Running](#3-running-an-agent) · [Sessions](#4-sessions) · [Limits and retries](#5-limits-and-retries) · [Approval](#6-human-approval) · [Hooks](#7-hooks) · [Context](#8-context-strategies) · [Multi-agent](#9-multi-agent) · [Tracing](#10-tracing) · [Testing](#11-testing-your-agents) · [Claude provider](#12-the-claude-provider) · [Troubleshooting](#13-troubleshooting)

---

## 1. First agent

An agent is a provider (the model), instructions (the system prompt), and tools.

```python
from agentloom import Agent, tool
from agentloom.providers.anthropic import AnthropicProvider

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

agent = Agent(
    AnthropicProvider(),
    instructions="You are a careful calculator. Use the tools for arithmetic.",
    tools=[add],
)

result = agent.run_sync("What is 1234 + 5678?")
print(result.output)   # the model's final text
print(result.status)   # RunStatus.COMPLETED
print(result.usage)    # tokens used across all steps
```

An `Agent` object holds configuration only. It keeps no conversation state, so one instance can serve many concurrent runs.

---

## 2. Tools

### Defining a tool

Decorate a function with `@tool`. The function's name, type hints and docstring become what the model sees, so write them for the model.

```python
from typing import Literal
from agentloom import tool

@tool
def search_orders(
    customer_email: str,
    status: Literal["open", "shipped", "cancelled"] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Find a customer's orders, newest first.

    Use this when the user asks about their orders but gives no order number.

    Args:
        customer_email: The email address on the account.
        status: Only return orders in this state. Omit for all orders.
        limit: Maximum number of orders to return (1-50).
    """
    ...
```

- The text before `Args:` becomes the tool description. Say what the tool does and when to use it.
- Each `Args:` entry becomes that parameter's description.
- Parameters without a default are required.

### Supported parameter types

| Python annotation | JSON Schema |
|---|---|
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `list[T]`, `set[T]`, `tuple[T, ...]` | `array` of `T` |
| `dict[str, T]` | `object` with `T` values |
| `T \| None`, `Optional[T]`, `A \| B` | `anyOf` |
| `Literal["a", "b"]` | `enum` |
| `Enum` subclass | `enum` of the values; your function receives the member |
| dataclass | `object`; your function receives an instance |
| pydantic v2 model | the model's schema; your function receives a validated instance |
| `Annotated[T, "text"]` | `T` with that description |

An unsupported annotation, a missing description, `*args`/`**kwargs`, or a duplicate tool name raises `ConfigurationError` when the tool is defined, not in the middle of a run.

### Validation and errors

Before your function runs, the arguments are checked against the schema. Problems are returned to the model as an error result so it can correct the call:

```
Invalid arguments:
- $.limit: expected integer, got str
```

Inside a tool, raise `ToolError` for failures the model can act on. The message is sent to the model, so write it as guidance:

```python
from agentloom import ToolError

@tool
def get_order(order_id: str) -> dict:
    """Fetch one order by its number."""
    order = db.get(order_id)
    if order is None:
        raise ToolError(f"No order {order_id!r}. Order numbers look like 'A-1001'.")
    return order
```

Any other exception is also caught, logged with its traceback under the `agentloom` logger, and reported to the model as `ExceptionType: message`. A tool failure never crashes a run.

### Return values

Return a string, or anything JSON-serialisable (dicts, lists, numbers, dataclasses, pydantic models). `None` is reported as `OK`. Output longer than `RunLimits.max_tool_output_chars` (100,000 by default) is cut, and the model is told how many characters were removed so it can ask for a narrower result.

### Sync and async

Both work. Sync functions run in a worker thread, so a slow database call cannot block the event loop. Use `async def` for I/O that has an async client.

### Tool options

```python
@tool(requires_approval=True, parallel=False, timeout=30)
def refund(order_id: str, amount_cents: int) -> str:
    """Refund part of an order to the original payment method."""
    ...
```

| Option | Default | Meaning |
|---|---|---|
| `name` | function name | Tool name shown to the model |
| `description` | docstring summary | Overrides the docstring |
| `requires_approval` | `False` | Every call goes through the agent's approval handler (see [section 6](#6-human-approval)) |
| `parallel` | `True` | Set `False` for tools with side effects. If any call in a step is non-parallel, that step's calls run one at a time, in order |
| `timeout` | `RunLimits.tool_timeout` | Seconds before the call is abandoned and reported as timed out |

A timed-out sync tool keeps running in its thread; Python cannot stop it. The model is told the operation may still have taken effect.

### Context, dependencies and per-run state

Add a parameter annotated `ToolContext` to receive run information. It is hidden from the model.

```python
from agentloom import ToolContext

@tool
async def my_orders(ctx: ToolContext) -> list[dict]:
    """List the current user's orders."""
    db, user_id = ctx.deps["db"], ctx.deps["user_id"]
    ctx.state["lookups"] = ctx.state.get("lookups", 0) + 1
    return await db.orders_for(user_id)

result = await agent.run("What did I order?", deps={"db": db, "user_id": "u_42"})
```

- `ctx.deps` is whatever you passed as `deps=` to `run`. Use it for database handles and the authenticated user. **Take identity from `deps`, never from model-supplied arguments**: the model can be talked into passing someone else's ID.
- `ctx.state` is a dict shared by all tool calls in one run.
- `ctx.run_id`, `ctx.agent_name` and `ctx.call` identify the call.

### Tools without the decorator

Construct a `Tool` directly when the schema comes from elsewhere (an OpenAPI document, an MCP server):

```python
from agentloom import Tool

async def handler(arguments, ctx):
    return await http.post("/search", json=dict(arguments))

search = Tool(
    name="search",
    description="Full-text search over the knowledge base.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=handler,
)
```

---

## 3. Running an agent

### Three ways to run

```python
result = await agent.run("question")          # async, returns RunResult
result = agent.run_sync("question")           # blocking, for scripts
async for event in agent.stream("question"):  # async, yields events as they happen
    ...
```

`run_sync` raises `ConfigurationError` if called inside a running event loop (for example in a Jupyter cell or a FastAPI handler); use `await agent.run(...)` there.

### The result

| Field | Meaning |
|---|---|
| `status` | How the run ended (table below) |
| `ok` | `True` only when `status` is `COMPLETED` |
| `output` | Text of the last assistant message of this run; may be empty |
| `messages` | Full history including earlier session turns |
| `new_messages` | What this run added |
| `usage` | Token totals, including sub-agents |
| `steps` | Number of model calls |
| `run_id` | Identifier that also appears on every event |

### Run status

Expected ways for a run to end are reported as a status, not raised, so the history and partial output are always available.

| `RunStatus` | Meaning | Typical response |
|---|---|---|
| `COMPLETED` | The model finished its answer | Use `output` |
| `MAX_STEPS` | Hit `RunLimits.max_steps` | Show partial progress; continue with a new run on the same session if appropriate |
| `TOKEN_BUDGET` | Hit `RunLimits.max_total_tokens` | Same |
| `TIMEOUT` | Hit `RunLimits.max_seconds` | Same |
| `TRUNCATED` | The final answer hit the output-token cap | Raise `max_tokens` or ask the model to continue |
| `REFUSED` | The model declined the request | Tell the user; do not retry unchanged |
| `CONTEXT_OVERFLOW` | The conversation no longer fits the model's window | Use a [context strategy](#8-context-strategies) |

**Always check `result.ok` (or call `result.raise_for_status()`) before treating `output` as a finished answer.**

### What is raised

| Exception | When |
|---|---|
| `ProviderError` | A model call failed and retries were exhausted, or the error was not retryable (bad request, authentication). Steps completed before the failure are already saved to the session |
| `ConfigurationError` | Misuse: no input and no pending turn in the session, `run_sync` inside an event loop, a badly defined tool |
| `SessionError` | The session store could not load or save |
| Exceptions from your hooks or approver | Propagate unchanged |

### Streaming

```python
from agentloom import TextDelta, ToolCallStarted, ToolCallFinished, RetryScheduled, RunFinished

async for event in agent.stream("Plan a 3-day trip to Kyoto"):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStarted):
        print(f"\n[calling {event.call.name}]")
    elif isinstance(event, RetryScheduled):
        print("\n[retrying]")       # text streamed for the failed attempt is void
    elif isinstance(event, RunFinished):
        result = event.result       # always the last event
```

Events, in the order they can occur: `RunStarted`, then per step `StepStarted`, `ThinkingDelta`\*, `TextDelta`\*, `RetryScheduled`\*, `ModelResponded`, `ToolCallStarted`\*, `ToolCallFinished`\*, and finally `RunFinished`. Every event carries `run_id`.

---

## 4. Sessions

Without a session, each run starts from nothing. Pass a session to keep a conversation going:

```python
from agentloom import FileSession, InMemorySession

session = FileSession("data/sessions/user_42.json")   # or InMemorySession()

await agent.run("My name is Ada.", session=session)
result = await agent.run("What's my name?", session=session)
```

- `InMemorySession` lives as long as the object.
- `FileSession` stores one JSON file per conversation. Each save writes a temporary file, flushes it to disk, and renames it over the old one, so a crash cannot leave a half-written history.

Use one session per conversation, and do not run two agents on the same session at the same time: both would save the full history and the later save would win.

### Crash recovery

The history is saved after every model response and after every batch of tool results. If the process dies between those two points, the stored history ends with tool calls that have no results. On the next run this is repaired automatically: each unanswered call receives an error result telling the model the call was interrupted and may or may not have taken effect.

To continue an interrupted run without adding a new user message, omit the input:

```python
result = await agent.run(session=session)
```

Because an interrupted tool may have run, **make tools with side effects idempotent** (accept an idempotency key, or check current state first). The framework can tell the model a call was interrupted; it cannot undo it.

To store sessions in a database, see [Writing a session store](architecture.md#writing-a-session-store).

---

## 5. Limits and retries

```python
from agentloom import Agent, RunLimits, RetryPolicy

agent = Agent(
    provider,
    tools=[...],
    limits=RunLimits(
        max_steps=25,                 # model calls per run
        max_total_tokens=500_000,     # None = unlimited
        max_seconds=300,              # None = unlimited; checked between steps
        max_repeated_calls=5,         # identical tool calls allowed per run
        tool_timeout=120,             # default seconds per tool call
        max_parallel_tools=8,         # concurrent tool calls in one step
        max_tool_output_chars=100_000,
    ),
    retry=RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0),
)
```

The values shown are the defaults, except `max_total_tokens` and `max_seconds`, which default to `None`.

**Repeated calls.** A model that is stuck often makes the same call again and again. Once the same tool has been called with the same arguments `max_repeated_calls` times in a run, further identical calls are not executed; the model gets an error telling it to change approach. Raise the limit for tools that are polled on purpose.

**Budgets are checked between steps.** A step that is already in flight finishes, so a run can exceed `max_total_tokens` or `max_seconds` by one step.

**Retries.** A model call is retried when the provider marks the error retryable: rate limits, overload, 5xx, timeouts and connection failures. The delay is random between 0 and `base_delay * 2^(attempt-1)`, capped at `max_delay`; a server-supplied `retry-after` is used instead when present. `max_attempts=1` turns retries off. Each retry emits a `RetryScheduled` event.

---

## 6. Human approval

For actions that are costly or hard to undo, require a person to confirm:

```python
from agentloom import Agent, Approval, ToolCall, ToolContext, tool

@tool(requires_approval=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email from the user's account."""
    ...

async def approver(call: ToolCall, ctx: ToolContext) -> bool | Approval:
    if call.name == "send_email" and call.arguments["to"].endswith("@ourcompany.com"):
        return True                                   # internal mail: allow
    answer = await ask_the_user(f"Run {call.name} with {dict(call.arguments)}?")
    return Approval(answer.yes, reason=answer.comment)

agent = Agent(provider, tools=[send_email], approval=approver)
```

- The approver sees the real arguments and returns `True`, `False` or `Approval(approved, reason)`.
- On denial the tool does not run; the model is told the call was not approved, with your reason.
- **If a tool requires approval and the agent has no approver, the call is refused.** A missing approver never means "allow".
- Steps that contain an approval-gated call run one call at a time, so prompts do not overlap.

The approver is awaited inside the run. For approvals that take hours, do not hold the run open: let the approver deny with a reason like "queued for review", record the request yourself, and start a new run on the same session once a decision exists.

---

## 7. Hooks

Hooks observe a run and can intervene at three points. Subclass `Hooks` and override what you need.

```python
from dataclasses import replace
from agentloom import Hooks, ToolDenied

class Guard(Hooks):
    async def on_event(self, event):
        metrics.count(type(event).__name__)

    async def before_tool(self, call, ctx):
        if call.name == "run_sql" and "drop " in call.arguments["query"].lower():
            raise ToolDenied("DROP statements are not allowed.")

    async def after_tool(self, call, result, ctx):
        return replace(result, content=redact_secrets(result.content))

agent = Agent(provider, tools=[...], hooks=[Guard()])
```

| Method | Purpose | If it raises |
|---|---|---|
| `on_event(event)` | Observe everything | Logged and ignored: a broken observer cannot fail a run |
| `before_model(request)` | Return a modified `ModelRequest`, or `None` | The run fails |
| `before_tool(call, ctx)` | Raise `ToolDenied` to block a call | `ToolDenied` blocks that call; anything else fails the run |
| `after_tool(call, result, ctx)` | Return a replacement `ToolResult`, or `None` | The run fails |

Hooks run in the order given. Use `before_tool` for policy that must hold whatever the prompt says; instructions guide the model, code enforces the rule.

---

## 8. Context strategies

A context strategy chooses which part of the stored history each model call sees. The stored history itself is never modified.

```python
from agentloom import Agent, SlidingWindow, Summarizing

Agent(provider)                                                   # KeepAll, the default
Agent(provider, context=SlidingWindow(max_messages=80, keep=40))
Agent(provider, context=Summarizing(cheap_provider, max_messages=80, keep=40))
```

| Strategy | Behaviour | Use when |
|---|---|---|
| `KeepAll` | Sends everything | The conversation fits the context window. This is most cases with current models |
| `SlidingWindow` | Past `max_messages`, drops the oldest and keeps about `keep` | Old turns do not matter |
| `Summarizing` | Same cut points, but the dropped part is replaced by a model-written summary | Long tasks where earlier decisions matter |

**Stay with `KeepAll` until you need otherwise.** Sending an unchanged, growing history is what lets the provider's prompt cache work, and prompt caching is usually the largest cost saving available.

The two trimming strategies are built to limit the damage:

- **Cuts are chunky.** With `max_messages=80, keep=40` the view is cut every 40 messages, not on every turn, so the prefix sent to the model stays identical between cuts and stays cacheable.
- **Cuts land on safe boundaries.** A view always starts at a real user message, never between a tool call and its result. If no safe boundary exists, nothing is cut.
- **Reasoning blocks are dropped from edited views.** Some models sign their reasoning blocks against the exact conversation that produced them and reject them after the history is edited. When a strategy has dropped anything, the request is marked `history_edited` and the Claude provider leaves reasoning blocks out of replayed turns, keeping text and tool calls.

`Summarizing` makes one extra model call per cut and caches the summary in memory for the life of the strategy object, so give long-lived conversations a long-lived strategy instance. A small, fast model is a sensible choice for the summariser.

---

## 9. Multi-agent

Turn an agent into a tool another agent can call:

```python
researcher = Agent(
    AnthropicProvider(),
    name="researcher",
    instructions="Research the question using the tools. Reply with findings and sources.",
    tools=[web_search, fetch_page],
)

lead = Agent(
    AnthropicProvider(),
    instructions=(
        "Answer the user's question. Delegate fact-finding to the researcher. "
        "It cannot see this conversation, so give it complete, self-contained tasks."
    ),
    tools=[researcher.as_tool(description="Researches one question and reports findings.")],
)
```

- The sub-agent starts with an **empty context** and receives only the `task` string. That isolation is the point: its searching and reading stay out of the lead's context, and only its conclusion comes back.
- Several delegations requested in one step run in parallel.
- The sub-agent's token usage is added to the lead's `result.usage`.
- If the sub-agent does not complete (step limit, refusal), the lead receives an error result saying why, with any partial output.
- `deps` passes through to the sub-agent's tools.

Reach for a second agent when a subtask would flood the main context or when subtasks can run in parallel. For a fixed sequence of steps, a single agent, or plain code calling `agent.run` several times, is simpler and cheaper.

---

## 10. Tracing

```python
from agentloom import ConsoleTracer, JsonlTracer

agent = Agent(provider, tools=[...], hooks=[ConsoleTracer(), JsonlTracer("runs.jsonl")])
```

`ConsoleTracer` prints one line per action to stderr:

```
> run 18d8c15e (agent)
  step 1: tool_use [in 812 / out 64 / cached 0]
  -> order_status({"order_id": "A-1001"})
  ok order_status 0.01s: {"status": "shipped", "eta": "2026-10-14"}
  step 2: end_turn [in 901 / out 48 / cached 812]
= completed after 2 step(s), 2637 tokens
```

`JsonlTracer` appends every event except streaming deltas as one JSON object per line with a timestamp: enough to reconstruct what a run did and what each step cost. The file contains prompts and tool data, so protect it accordingly.

To send traces elsewhere (OpenTelemetry, a database), write a `Hooks` subclass with `on_event`; `agentloom.tracing.event_to_dict(event)` gives a JSON-friendly dict.

Framework warnings (retries, tool exceptions, failing hooks) go to the standard `logging` logger named `agentloom`.

---

## 11. Testing your agents

`ScriptedProvider` stands in for the model and replays a script, one entry per model call.

```python
from agentloom import Agent, RunStatus, RunLimits
from agentloom.testing import ScriptedProvider, call

async def test_refund_requires_lookup_first():
    model = ScriptedProvider([
        call("get_order", order_id="A-1"),             # step 1: the model calls a tool
        [call("refund", order_id="A-1", amount_cents=500),
         call("notify", order_id="A-1")],              # step 2: two calls at once
        "Refunded $5.00.",                             # step 3: final answer
    ])
    agent = Agent(model, tools=[get_order, refund, notify])

    result = await agent.run("Refund $5 on A-1")

    assert result.status is RunStatus.COMPLETED
    sent_back = model.requests[1].messages[-1].tool_results   # what the model saw
    assert not sent_back[0].is_error
```

A script entry can be a string (final answer), a `ToolCall` or list of them, a `ModelResponse` (to control the stop reason or usage), an exception instance (raised, for testing retries), or a function `request -> entry`.

`model.requests` records every request, so you can assert on what the agent sent. If the agent calls the model more times than the script has entries, the provider raises, which catches unexpected loops.

This tests your tools, limits, hooks and approval logic deterministically. It does not test how a real model behaves with your prompts; that needs evaluation runs against the real model.

---

## 12. The Claude provider

```python
from agentloom.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    "claude-opus-5-5",        # default
    max_tokens=64_000,        # default output cap per call
    effort=None,              # "low" | "medium" | "high" | "xhigh" | "max"; None = model default
    thinking=None,            # e.g. {"type": "adaptive", "display": "summarized"}
    cache=True,               # automatic prompt caching
    refusal_fallback=True,    # see below
    eager_tool_streaming=True,
    betas=(),
    extra=None,               # any other Messages API parameters
)
```

What it does for you:

- **Always streams**, so long outputs do not hit HTTP timeouts.
- **Replays the model's own content blocks unchanged** on later turns, including reasoning blocks, as the API requires.
- **Turns on prompt caching** with a top-level `cache_control`. Keep your instructions and tool list stable within a session so the cache can hit; check `result.usage.cache_read_tokens`.
- **Creates its client with SDK retries off**, because the agent's `RetryPolicy` handles retries and reports them as events. If you pass your own client, set `max_retries=0` on it to avoid retrying twice.

To see reasoning as `ThinkingDelta` events, pass `thinking={"type": "adaptive", "display": "summarized"}`; by default current models return reasoning blocks with empty text.

**Refusal fallback.** Some current Claude models run safety classifiers that can decline a request. With `refusal_fallback=True` (the default) the provider asks the API to re-run a declined request on a fallback model, using the beta parameter `fallbacks="default"`. It applies only to `claude-fable-5-1`, `claude-opus-5-5`, `claude-opus-5` and `claude-sonnet-5-5`, and it routes the call through the SDK's beta endpoint. **Set `refusal_fallback=False`** if you use Amazon Bedrock, Vertex AI, Microsoft Foundry or a proxy that rejects the parameter, or if you do not want requests served by another model. Refusals then end the run with `RunStatus.REFUSED`.

**Other platforms.** Pass a platform client from the Anthropic SDK:

```python
from anthropic import AsyncAnthropicVertex

provider = AnthropicProvider(
    "claude-opus-5-5",
    client=AsyncAnthropicVertex(project_id="my-project", region="global", max_retries=0),
    refusal_fallback=False,
)
```

**Inspecting requests.** `provider.build_params(request)` returns the exact keyword arguments that would be sent, which is useful in tests and when debugging.

For models from other vendors, see [Writing a provider](architecture.md#writing-a-provider).

---

## 13. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ConfigurationError: AnthropicProvider needs the Anthropic SDK` | `pip install "agentloom[anthropic]"` |
| `FileNotFoundError` from `ssl.create_default_context` when creating the provider | The `SSL_CERT_FILE` environment variable points to a file that does not exist. Unset it or point it at a real CA bundle |
| `ProviderError: 401` | Missing or wrong API key |
| `ProviderError: 400` mentioning `fallbacks` or an unknown beta | Your endpoint does not accept the fallback beta: `AnthropicProvider(refusal_fallback=False)` |
| `ProviderError: 400` mentioning a `signature` in a `thinking` block | The history sent to the model was edited (a `before_model` hook, a custom context strategy, or changed instructions/tools mid-session). Keep the history append-only, or set `history_edited=True` on the request so reasoning blocks are dropped |
| The run ends with `MAX_STEPS` | The task needs more steps (raise `max_steps`) or the model is looping (read the trace) |
| The model keeps getting "Invalid arguments" | The tool's description or parameter docs are unclear. Improve the docstring before changing anything else |
| `ConfigurationError: run_sync() was called from a running event loop` | Use `await agent.run(...)` |
| `cache_read_tokens` is always 0 | Something in the prefix changes each call: a timestamp in `instructions`, a tool list that varies, or a trimming context strategy |
