# agentloom

**English** | [简体中文](README.zh-CN.md)

A small, dependable Python framework for building tool-using LLM agents.

agentloom gives you the agent loop and the things that make it survive production: validated tool calls, retries, hard limits, crash-safe sessions, human approval, and a full event stream. The core has no dependencies, and the whole package is about 2,300 lines you can read in an afternoon.

```python
from agentloom import Agent, tool
from agentloom.providers.anthropic import AnthropicProvider

@tool
def order_status(order_id: str) -> dict:
    """Look up an order's status, items and estimated delivery date."""
    return {"status": "shipped", "eta": "2026-10-14"}

agent = Agent(
    AnthropicProvider(),
    instructions="You are a concise customer-support assistant.",
    tools=[order_status],
)
result = agent.run_sync("Where is my order A-1001?")
print(result.output)
```

## Install

Requires Python 3.10 or newer.

```bash
pip install -e ".[anthropic]"     # from this directory; includes the Claude adapter
pip install -e ".[dev]"           # plus test and lint tools
```

Set `ANTHROPIC_API_KEY` (or any other credential the Anthropic SDK reads) before using `AnthropicProvider`.

To see the framework work without an API key:

```bash
python examples/03_offline_demo.py
```

## What you get

| Area | What it does |
|---|---|
| **Tools** | `@tool` turns a typed function into a tool: JSON Schema from the signature, description from the docstring. Arguments are validated before your code runs, and failures go back to the model as errors it can fix. |
| **Agent loop** | Streaming model calls, parallel tool execution, one result message per step, correct handling of truncation, pauses and refusals. |
| **Limits** | Caps on steps, tokens, wall-clock time, tool duration, tool output size, and repeated identical calls. A run always terminates. |
| **Retries** | Exponential backoff with jitter for retryable provider errors, honouring `retry-after`. |
| **Sessions** | Conversation history saved after every step with atomic writes. A run cut off mid-step can be resumed. |
| **Approval** | Mark a tool `requires_approval=True` and every call waits for your approver. No approver means the call is refused. |
| **Hooks and events** | Observe every step, rewrite requests, block or redact tool calls. Console and JSONL tracers are included. |
| **Context strategies** | Keep everything (default), a sliding window, or model-written summaries, all with cache-friendly cut points. |
| **Multi-agent** | `agent.as_tool()` lets one agent delegate to another with an isolated context. |
| **Testing** | `ScriptedProvider` replays a script in place of a model so agent behaviour is unit-testable. |

## Documentation

- [User guide](docs/en/guide.md): every feature, with examples
- [Architecture and extension](docs/en/architecture.md): design, guarantees, and how to write your own provider, session store, context strategy or hook
- [Examples](examples/): quickstart, interactive chat, offline demo

## Status

Version 0.1.0. What has and has not been verified:

- **Verified:** 67 tests pass, covering the loop, tools, sessions, context strategies, tracing and the Claude adapter. `mypy --strict` and `ruff` are clean. The Claude adapter tests run the real Anthropic SDK (1.13) against a local fake HTTP server, so request construction and stream parsing are exercised end to end.
- **Not verified:** No test has called the live Claude API. Behaviour that only the real service can confirm (acceptance of the `fallbacks` beta parameter on your account, prompt-cache hit rates, reasoning-block replay rules) should be checked with a real key before you rely on it.
- **Included providers:** Claude through the official Anthropic SDK. Other model APIs need a provider class of your own; the [architecture guide](docs/en/architecture.md#writing-a-provider) shows how, and it is about 60 lines.
- **Not included:** structured (schema-constrained) final output, MCP client support, built-in file/shell/web tools, a durable workflow engine. See the [roadmap](docs/en/architecture.md#what-is-not-here).

## Development

```bash
pytest          # tests
mypy            # strict type check
ruff check .    # lint
ruff format .   # format
```

## License

MIT
