# agentloom

[English](README.md) | **简体中文**

一个小而可靠的 Python 框架，用来构建会调用工具的 LLM Agent。

agentloom 提供 Agent 循环，以及让它能在生产环境里跑得住的配套能力：工具参数校验、重试、硬性上限、崩溃可恢复的会话、人工审批，以及完整的事件流。核心零依赖，整个包约 2300 行，一个下午可以读完。

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

## 安装

需要 Python 3.10 或更高版本。

```bash
pip install -e ".[anthropic]"     # 在本目录执行；包含 Claude 适配器
pip install -e ".[dev]"           # 另含测试与代码检查工具
```

使用 `AnthropicProvider` 前，请设置 `ANTHROPIC_API_KEY`（或 Anthropic SDK 支持的其他凭据）。

不需要 API Key 也能看到框架的实际运行效果：

```bash
python examples/03_offline_demo.py
```

## 功能一览

| 模块 | 作用 |
|---|---|
| **工具** | `@tool` 把带类型注解的函数变成工具：由函数签名生成 JSON Schema，由文档字符串生成描述。参数在进入你的代码之前先校验，校验失败会作为错误返回给模型，让它自行修正。 |
| **Agent 循环** | 流式调用模型、并行执行工具、每步结果合并为一条消息，并正确处理输出截断、暂停和拒答。 |
| **上限** | 步数、token、总耗时、单次工具耗时、工具输出长度、重复调用次数都有上限，一次运行必然会结束。 |
| **重试** | 对可重试的模型调用错误做带抖动的指数退避，并遵循服务端的 `retry-after`。 |
| **会话** | 每一步之后以原子写入方式保存对话历史，运行中途被打断后可以续跑。 |
| **审批** | 给工具加上 `requires_approval=True`，每次调用都要先经过你的审批函数；没有配置审批函数时一律拒绝。 |
| **钩子与事件** | 观察每一步、改写请求、拦截或脱敏工具调用。内置控制台和 JSONL 两种追踪器。 |
| **上下文策略** | 全量保留（默认）、滑动窗口、模型摘要三种，裁剪点对提示缓存友好。 |
| **多 Agent** | `agent.as_tool()` 让一个 Agent 把任务委派给另一个上下文相互隔离的 Agent。 |
| **测试** | `ScriptedProvider` 按脚本代替模型作答，Agent 的行为可以写成单元测试。 |

## 文档

- [使用指南](docs/zh/guide.md)：逐项讲解全部功能，附示例
- [架构与扩展](docs/zh/architecture.md)：设计思路、可靠性保证，以及如何自己实现 Provider、会话存储、上下文策略和钩子
- [示例](examples/)：快速上手、交互式聊天、离线演示

## 当前状态

版本 0.1.0。下面如实说明哪些已经验证、哪些还没有：

- **已验证：** 67 个测试全部通过，覆盖 Agent 循环、工具、会话、上下文策略、追踪和 Claude 适配器；`mypy --strict` 与 `ruff` 均无告警。Claude 适配器的测试使用真实的 Anthropic SDK（1.13）对接本地伪造的 HTTP 服务，因此请求构造和流式解析都走了完整链路。
- **未验证：** 没有任何测试调用过真实的 Claude API。只有真实服务才能确认的行为（你的账号是否接受 `fallbacks` 这个 beta 参数、提示缓存的实际命中率、推理块回传规则等），在依赖它之前请用真实 Key 自行验证。
- **内置 Provider：** 只有通过官方 Anthropic SDK 接入的 Claude。其他模型需要自己写一个 Provider 类，[架构文档](docs/zh/architecture.md#编写-provider)里有写法，大约 60 行。
- **未包含：** 结构化（受 Schema 约束的）最终输出、MCP 客户端、内置的文件/Shell/网页工具、持久化工作流引擎。详见[未包含的功能](docs/zh/architecture.md#未包含的功能)。

## 开发

```bash
pytest          # 运行测试
mypy            # 严格类型检查
ruff check .    # 代码检查
ruff format .   # 格式化
```

## 许可证

MIT
