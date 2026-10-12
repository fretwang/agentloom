# Contributing to agentloom

Thanks for your interest. 中文说明见[下方](#中文说明)。

## Setup

```bash
git clone https://github.com/fretwang/agentloom && cd agentloom
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Before opening a pull request, run what CI runs:

```bash
ruff check . && ruff format --check .
mypy
pytest
```

No API key is needed for any of it. `python examples/03_offline_demo.py` shows the loop running against a scripted model.

## What makes a change easy to merge

- **It keeps the core small.** The core (`agent.py`, `tools.py`, `session.py`, `context.py`) has no third-party imports and should stay that way. Anything that needs an SDK goes in its own module behind an optional extra, the way `providers/` and `mcp.py` do.
- **It comes with a test that would fail without it.** Agent behaviour is tested with `ScriptedProvider`; adapters are tested by running the real SDK against a local fake (see `tests/test_openai_provider.py` and `tests/test_mcp.py`). Tests must not call a live API.
- **It is honest about what was verified.** If something was only exercised against a fake, say so in the pull request and, where users would rely on it, in the docs.
- **It updates both languages.** User-facing docs exist in `docs/en` and `docs/zh`, and the README in both. If you only write one of the two, say so and a maintainer will help with the other.
- **Failures the model can fix go to the model.** A bad argument or a failed tool becomes an error result, not an exception. See the design principles in [docs/en/architecture.md](docs/en/architecture.md).

## Good first contributions

Each of these is self-contained and has an existing piece of code to model it on. Open an issue first if you want to check the approach.

| Idea | Start from | Notes |
|---|---|---|
| A provider for another model API (Gemini, Bedrock Converse, ...) | `providers/openai.py`, [Writing a provider](docs/en/architecture.md#writing-a-provider) | About 60 to 300 lines plus tests against a fake HTTP transport |
| A Redis or SQLite session store | `FileSession` in `session.py`, [Writing a session store](docs/en/architecture.md#writing-a-session-store) | `save` must replace the history atomically |
| A context strategy that drops old tool output | `SlidingWindow` in `context.py` | Return `edited=True` whenever the view differs from the input |
| An OpenTelemetry tracer | `JsonlTracer` in `tracing.py` | A `Hooks` subclass; keep the dependency optional |
| A verified entry for an OpenAI-compatible service | [the endpoint table](docs/en/guide.md#13-the-openai-compatible-provider) | Run `examples/04_openai_compatible.py` against the service and report what worked, including the model ID and any option you had to change |
| Images in messages | `types.py` and both providers | A new block type plus encoding in each provider |
| Another template in `examples/` | `examples/06_support_bot.py` | Expose `build_agent(provider)` and add a scripted test in `tests/test_examples.py` |

## Reporting bugs

Use the bug report template. The most useful thing you can attach is a trace: add `hooks=[JsonlTracer("run.jsonl")]` to the agent and include the file with secrets removed. It contains prompts and tool data, so read it before you share it.

Please report security problems privately through GitHub's "Report a vulnerability" button on the repository's Security tab, not in a public issue.

## 中文说明

环境搭建和提交前的检查命令同上，全部不需要 API Key。

容易被合并的改动通常满足这几点：

- **不让核心变大。** 核心模块（`agent.py`、`tools.py`、`session.py`、`context.py`）不导入任何第三方包。需要 SDK 的功能请放在独立模块里，并做成可选依赖，参照 `providers/` 和 `mcp.py`。
- **带一个没有这次改动就会失败的测试。** Agent 的行为用 `ScriptedProvider` 测试；适配器的测试让真实 SDK 对接本地替身（参见 `tests/test_openai_provider.py` 和 `tests/test_mcp.py`）。测试不得调用真实 API。
- **如实说明验证到了什么程度。** 如果只在替身上验证过，请在 PR 里写明；用户会依赖的行为还要写进文档。
- **中英文文档一起更新。** `docs/en` 与 `docs/zh`、两份 README 需要保持一致。只写了一种语言也没关系，在 PR 里说明，维护者会协助补全。
- **模型能自己修正的失败，交还给模型。** 参数错误或工具失败应当变成错误结果，而不是异常。设计原则见 [docs/zh/architecture.md](docs/zh/architecture.md)。

适合新手的贡献见上方表格：新的 Provider、Redis 或 SQLite 会话存储、新的上下文策略、OpenTelemetry 追踪器、为某个 OpenAI 兼容服务提交实测结果、消息中的图片支持、新的示例模板。动手之前可以先开一个 issue 确认思路。

报告 Bug 时请使用模板，并尽量附上追踪文件（`hooks=[JsonlTracer("run.jsonl")]`）。文件里包含提示词和工具数据，分享前请先检查并去掉敏感信息。安全问题请通过仓库 Security 页的 "Report a vulnerability" 私下报告。
