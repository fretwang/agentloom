# agentloom

**English** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/fretwang/agentloom/actions/workflows/ci.yml/badge.svg)](https://github.com/fretwang/agentloom/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**The agent loop you can read in an afternoon.** A small Python framework for tool-using LLM agents: about 2,900 lines, no dependencies in the core, and every run is guaranteed to end.

It works with Claude, with any OpenAI-compatible API (OpenAI, DeepSeek, Qwen, Moonshot, OpenRouter, Ollama, vLLM), and with tools from any MCP server.

<p align="center">
  <img src="docs/assets/demo.svg" alt="Output of the offline demo: two parallel tool calls, a validation error the model corrects, a sub-agent, and the final answer" width="682">
</p>

That is the real output of `python examples/03_offline_demo.py`, which needs no API key. A scripted stand-in plays the model; everything else is the machinery a real run uses. A bad argument (`42` instead of a string) never reaches your code: it goes back to the model as an error it can fix.

## Quickstart

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

Another model is a one-line change:

```python
from agentloom.providers.openai import OpenAIProvider

provider = OpenAIProvider("deepseek-chat", base_url="https://api.deepseek.com")   # DeepSeek
provider = OpenAIProvider("llama3.3", base_url="http://localhost:11434/v1", api_key="ollama")  # Ollama
```

Ask for a typed answer instead of text, and plug in an MCP server:

```python
from agentloom.mcp import MCPTools

@dataclass
class Review:
    verdict: Literal["approve", "request_changes"]
    findings: list[str]

async with MCPTools.stdio("npx", "-y", "@modelcontextprotocol/server-filesystem", ".") as files:
    agent = Agent(provider, tools=files, output_type=Review)
    review = (await agent.run("Review the Python files here.")).data   # a Review instance
```

## Install

Requires Python 3.10 or newer. agentloom is not on PyPI yet; install it from GitHub or from a clone:

```bash
pip install "agentloom[anthropic] @ git+https://github.com/fretwang/agentloom"   # Claude
pip install "agentloom[openai] @ git+https://github.com/fretwang/agentloom"      # OpenAI-compatible APIs
pip install "agentloom[mcp] @ git+https://github.com/fretwang/agentloom"         # MCP servers

git clone https://github.com/fretwang/agentloom && cd agentloom
pip install -e ".[dev]"            # everything, plus test and lint tools
python examples/03_offline_demo.py
```

`AnthropicProvider` reads `ANTHROPIC_API_KEY` (or any other credential the Anthropic SDK reads). `OpenAIProvider` reads `OPENAI_API_KEY` unless you pass `api_key`.

## What you get

| Area | What it does |
|---|---|
| **Tools** | `@tool` turns a typed function into a tool: JSON Schema from the signature, description from the docstring. Arguments are validated before your code runs, and failures go back to the model as errors it can fix. |
| **Agent loop** | Streaming model calls, parallel tool execution, one result message per step, correct handling of truncation, pauses and refusals. |
| **Limits** | Caps on steps, tokens, wall-clock time, tool duration, tool output size, and repeated identical calls. A run always terminates. |
| **Retries** | Exponential backoff with jitter for retryable provider errors, honouring `retry-after`. |
| **Sessions** | Conversation history saved after every step with atomic writes. A run cut off mid-step can be resumed. |
| **Approval** | Mark a tool `requires_approval=True` and every call waits for your approver. No approver means the call is refused. |
| **Structured output** | `output_type=` a dataclass, pydantic model or JSON Schema gives you a validated object in `result.data`. Invalid answers go back to the model to be fixed. |
| **MCP** | `MCPTools` connects to an MCP server over stdio or HTTP and exposes its tools as ordinary tools, so limits, approval and tracing apply to them too. |
| **Hooks and events** | Observe every step, rewrite requests, block or redact tool calls. Console and JSONL tracers are included. |
| **Context strategies** | Keep everything (default), a sliding window, or model-written summaries, all with cache-friendly cut points. |
| **Multi-agent** | `agent.as_tool()` lets one agent delegate to another with an isolated context. |
| **Providers** | Claude through the Anthropic SDK, and any OpenAI-compatible Chat Completions API through the OpenAI SDK. A new provider is one small class. |
| **Testing** | `ScriptedProvider` replays a script in place of a model so agent behaviour is unit-testable. |

## When to use it, and when not to

agentloom is small on purpose. There is no graph engine, no planner abstraction and no hidden prompt: the model sees your instructions, your tool descriptions and the conversation, and the whole loop is one function you can step through in a debugger.

**It fits when** you want to understand and own the loop your product depends on, you care about what happens on the bad days (timeouts, crashes mid-step, malformed tool calls, runaway loops), and you would rather add the few integrations you need than carry many you do not.

**Reach for something else when** you need what a larger framework bundles:

- graph-shaped workflows or durable multi-day execution: LangGraph, or a workflow engine
- a large catalogue of ready-made integrations (vector stores, document loaders, hosted tools): LangChain
- Pydantic-first typing throughout, with built-in evaluation tooling: PydanticAI
- handoffs, guardrails and hosted tracing on OpenAI's platform: the OpenAI Agents SDK

## Templates

Starting points you can copy. Examples 05 to 08 run on Claude or any OpenAI-compatible model, and each of them is covered by a test that drives it with a scripted model.

| Example | What it shows |
|---|---|
| [`03_offline_demo.py`](examples/03_offline_demo.py) | The loop with no API key: validation, error feedback, parallel calls, a sub-agent |
| [`02_chat.py`](examples/02_chat.py) | Streaming chat with a saved session and human approval |
| [`05_mcp.py`](examples/05_mcp.py) | Tools from an MCP server, with approval on the ones that change things |
| [`06_support_bot.py`](examples/06_support_bot.py) | Customer support: identity from `deps`, refund policy as an approver, audit log |
| [`07_code_review.py`](examples/07_code_review.py) | Reviews a git diff and returns a typed `Review` object |
| [`08_docs_qa.py`](examples/08_docs_qa.py) | Question answering over your Markdown docs (RAG) with citations, no extra dependencies |

## Documentation

- [User guide](docs/en/guide.md): every feature, with examples
- [Architecture and extension](docs/en/architecture.md): design, guarantees, and how to write your own provider, session store, context strategy or hook

## Status

Version 0.1.0, alpha. 115 tests pass, and `mypy --strict` and `ruff` are clean. The provider and MCP tests run the real SDKs against local fakes, but **no test calls a live model API**: check the behaviour that only a real service can confirm before you rely on it. The details are in [verification status](docs/en/architecture.md#verification-status).

## Contributing

Bug reports, new providers and session stores are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains the setup and lists good first contributions.

```bash
pytest          # tests
mypy            # strict type check
ruff check .    # lint
ruff format .   # format
```

## License

MIT
