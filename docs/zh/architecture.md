# 架构与扩展

[English](../en/architecture.md)

目录：[设计原则](#设计原则) · [模块结构](#模块结构) · [一次运行的全过程](#一次运行的全过程) · [可靠性保证](#可靠性保证) · [编写 Provider](#编写-provider) · [编写会话存储](#编写会话存储) · [编写上下文策略](#编写上下文策略) · [未包含的功能](#未包含的功能)

---

## 设计原则

**1. 只有一个循环，而且看得见。** Agent 循环就是 `Agent._iterate` 这一个函数，约 120 行。没有图引擎，没有规划器抽象，也没有隐藏的提示词：模型收到的文字只有你写的 instructions、你的工具描述，以及对话本身。

**2. 四个扩展点，每个都是很小的接口。** 不同部署之间会变化的东西，都放在四个接口之一的后面，每个接口只有一两个方法：

| 扩展点 | 接口 | 方法 | 内置实现 |
|---|---|---|---|
| 模型 API | `Provider` | `stream` | `AnthropicProvider`、`ScriptedProvider` |
| 持久化 | `Session` | `load`、`save` | `InMemorySession`、`FileSession` |
| 上下文选择 | `ContextStrategy` | `prepare` | `KeepAll`、`SlidingWindow`、`Summarizing` |
| 策略与观察 | `Hooks` | `on_event`、`before_model`、`before_tool`、`after_tool` | `ConsoleTracer`、`JsonlTracer` |

核心不导入任何第三方包。Anthropic SDK 只在 `AnthropicProvider` 内部导入。

**3. 模型能自己纠正的失败交给模型，其余的明确告诉调用方。** 参数错误、工具不存在、工具抛异常、超时、审批被拒，都会变成对话里的一条错误结果，因为模型往往能自行恢复。需要调用方处理的情况（达到上限、拒答、截断）以 `RunStatus` 表示。只有基础设施故障和编程错误才会抛异常。

**4. 默认安全。** 没有审批函数就拒绝；被截断的工具参数不执行；每次运行都有步数上限；工具参数总是先校验。安全措施是需要你主动关闭的，而不是需要你主动开启的。

**5. 历史只追加。** 已存的对话只会增长。上下文策略产出的是视图，从不改写已存内容。这样提示缓存能持续命中，带签名的推理块保持有效，会话本身也是一份忠实的审计记录。

---

## 模块结构

```
src/agentloom/
├── types.py        Message、TextBlock、ToolCall、ToolResult、Usage、ModelRequest、ModelResponse
├── errors.py       异常体系
├── tools.py        @tool、Tool、Toolset、ToolContext、Schema 生成与校验
├── policies.py     RetryPolicy、RunLimits
├── session.py      Session、InMemorySession、FileSession、repair_history
├── context.py      ContextStrategy、KeepAll、SlidingWindow、Summarizing
├── events.py       各事件的 dataclass
├── hooks.py        Hooks 基类
├── agent.py        Agent、RunResult、RunStatus、Approval：循环本体
├── tracing.py      ConsoleTracer、JsonlTracer
├── testing.py      ScriptedProvider、call
└── providers/
    ├── base.py       Provider、TextChunk、ThinkingChunk、Completed
    └── anthropic.py  AnthropicProvider
```

依赖方向是单向的：`agent.py` 依赖其余所有模块；除了 `events.py` 为了一处类型注解之外，没有模块依赖 `agent.py`。

### 数据模型

`Message` 有一个 `role`（`"user"` 或 `"assistant"`）和一个内容块列表：

- `TextBlock(text)`
- `ToolCall(id, name, arguments)`，出现在助手消息中
- `ToolResult(tool_call_id, content, is_error)`，出现在用户消息中

工具、钩子、会话和上下文策略操作的都是这套与厂商无关的视图。消息还可以带一个 `provider_data`：产生这条消息的 Provider 的原生数据，只有该 Provider 自己会读取。Claude 的推理块就是这样原样回传的，框架其余部分完全不需要知道它们的存在。`provider_data` 必须能序列化为 JSON，这样会话才能保存它。

---

## 一次运行的全过程

```
agent.run(input, session, deps)
│
├─ 从会话加载历史；如果上次运行被打断，修复末尾悬空的工具调用
├─ 追加用户消息；保存
│
└─ 循环
   ├─ 达到上限？ ─────────────────────────► 状态 MAX_STEPS / TOKEN_BUDGET / TIMEOUT
   ├─ context.prepare(history) ─► 视图
   ├─ hooks.before_model(request)
   ├─ provider.stream(request)            （可重试的 ProviderError：退避后重试）
   │     └─ 发出 TextDelta / ThinkingDelta
   ├─ 追加助手消息；保存
   ├─ 停止原因：
   │     拒答 ────────────────────────────► 状态 REFUSED
   │     上下文溢出 ──────────────────────► 状态 CONTEXT_OVERFLOW
   │     暂停 ────────────────────────────► 进入下一轮（把历史原样再发一次）
   │     没有工具调用 ────────────────────► 状态 COMPLETED（或 TRUNCATED）
   ├─ 对每个工具调用（所有相关工具都允许时并行执行）：
   │     工具不存在？          ─► 错误结果
   │     重复次数过多？        ─► 错误结果
   │     参数不合法？          ─► 错误结果
   │     需要审批？            ─► 审批函数；被拒 ─► 错误结果
   │     hooks.before_tool     ─► ToolDenied ─► 错误结果
   │     带超时执行处理函数    ─► 结果，或错误结果
   │     hooks.after_tool
   └─ 追加【一条】包含全部结果的用户消息；保存
```

有两处细节关系到正确性：

- **同一步的所有结果放在一条消息里按调用顺序返回。** 拆成多条消息虽然模型 API 也接受，但会让模型逐渐不再发起并行调用。
- **如果响应在请求工具时被输出上限截断，这些调用不会执行。** 只生成了一半的参数有可能恰好是合法 JSON，只是值被截短了。框架会给每个调用返回一条错误，要求模型用更小的参数重试。

---

## 可靠性保证

下表列出框架保证了什么，以及保证的边界在哪里。

| 保证 | 实现机制 | 边界 |
|---|---|---|
| 运行必然结束 | 始终设有 `max_steps`；可选的 token 与时间预算 | 单次模型调用或工具调用受各自超时约束，不受运行预算约束 |
| 工具代码不会收到违反 Schema 的参数 | 每次调用前执行 `validate_json`；pydantic 与 dataclass 转换 | 语义层面的检查（这个 ID 是否属于当前用户）是工具自己的职责 |
| 工具失败不会让运行崩溃 | 处理函数抛出的所有异常都转为错误结果 | `BaseException`（任务取消、`KeyboardInterrupt`）会继续向上传播 |
| 已存的历史总能再次发给模型 | 每半步保存一次；加载时修复悬空的工具调用 | 前提是每个会话只有一个写入者 |
| 保存的会话文件不会是半截的 | 临时文件、`fsync`、原子重命名 | 仅限 `FileSession`；其他存储需自行保证原子性 |
| 崩溃时正在执行的工具调用不会被悄悄重跑 | 被打断的调用得到一条错误结果，由模型决定下一步 | 框架无法知道副作用是否已经发生：请把工具做成幂等的 |
| 需要审批的工具未经批准不会执行 | 在循环中检查，早于 `before_tool` 钩子 | 审批函数自身的逻辑 |
| 观察者出错不会让运行失败 | `on_event` 的异常被记录并忽略 | `before_model`、`before_tool`、`after_tool` 是控制点，其异常会向上传播 |
| 重试有上限且可见 | `RetryPolicy.max_attempts`；`RetryScheduled` 事件 | 失败尝试期间已经流式输出的文字已到达你的界面，请在收到 `RetryScheduled` 时清除 |

### 不保证的事情

- **工具恰好执行一次。** 一般情况下做不到。框架提供的是：每个已记录的调用至多执行一次，并在崩溃后如实给出"这次调用可能已经发生"的信号。
- **防住提示注入。** 工具结果是数据，但模型仍可能被其中的文字误导。请限制工具的能力，从 `deps` 获取身份，对有实际后果的操作启用审批。不要指望靠提示词来保证安全。
- **会话的并发控制。** 两个运行同时使用一个会话会互相覆盖。请在应用层按对话串行化。

---

## 编写 Provider

一个 Provider 适配一种模型 API。实现 `stream` 即可：先按需产出若干 `TextChunk`/`ThinkingChunk`，最后产出**恰好一个**带完整响应的 `Completed`。失败时抛出 `ProviderError` 并正确设置 `retryable`，不要在内部自行重试。

下面的骨架以一个假想的 JSON API 为例，覆盖了 Provider 的全部职责：

```python
from collections.abc import AsyncIterator

from agentloom import (
    Completed, Message, ModelRequest, ModelResponse, Provider, ProviderChunk, ProviderError,
    StopReason, TextBlock, TextChunk, ToolCall, ToolResult, Usage,
)


class AcmeProvider(Provider):
    name = "acme"   # 会记录在消息上，用来标识谁可以读取 provider_data

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
            reply = await self.client.chat(payload)            # 1. 调用 API
        except self.client.RateLimited as exc:                 # 2. 对失败分类
            raise ProviderError(str(exc), retryable=True, status_code=429,
                                retry_after=exc.retry_after) from exc
        except self.client.ServerError as exc:
            raise ProviderError(str(exc), retryable=True, status_code=exc.status) from exc
        except self.client.ClientError as exc:
            raise ProviderError(str(exc), retryable=False, status_code=exc.status) from exc

        blocks: list[TextBlock | ToolCall | ToolResult] = []   # 3. 解码为中立的内容块
        if reply["text"]:
            yield TextChunk(reply["text"])                     #    （流式 API 应边收边产出）
            blocks.append(TextBlock(reply["text"]))
        for item in reply.get("tool_calls", []):
            blocks.append(ToolCall(item["id"], item["name"], item["arguments"]))

        stop = {                                               # 4. 统一停止原因
            "stop": StopReason.END_TURN,
            "tool_calls": StopReason.TOOL_USE,
            "length": StopReason.MAX_TOKENS,
        }.get(reply["finish_reason"], StopReason.OTHER)

        yield Completed(ModelResponse(                         # 5. 以恰好一个 Completed 结束
            message=Message("assistant", blocks, provider=self.name),
            stop_reason=stop,
            usage=Usage(reply["usage"]["in"], reply["usage"]["out"]),
            model=self.model,
        ))

    def _encode(self, message: Message) -> dict:
        ...  # 把 TextBlock / ToolCall / ToolResult 映射成该 API 的消息格式
```

检查清单：

1. **停止原因。** 把"触及输出上限"映射为 `MAX_TOKENS`，把安全拒答映射为 `REFUSAL`。循环对"被截断的工具调用"的处理依赖于 `MAX_TOKENS` 被如实上报。
2. **工具调用的参数必须是字典。** 如果 API 以 JSON 字符串返回参数，请先解析；解析失败时抛出 `ProviderError(retryable=True)`，让这一轮重新发起。
3. **工具结果。** 有些 API 要求每个结果单独成一条 `tool` 角色的消息，请在 `_encode` 里把 `ToolResult` 块相应展开。框架交给你的总是"一步的所有结果在同一条用户消息里"。
4. **`provider_data`。** 如果 API 返回了必须原样回传的内容，请存入 `Message.provider_data`（需可序列化为 JSON），并设置 `provider=self.name`。当 `request.history_edited` 为真时，要去掉那些只在历史未被改动时才有效的内容。
5. **重试。** 关闭 SDK 自带的重试，否则会重试两遍。
6. **测试。** 参照 `tests/test_anthropic_provider.py` 的做法：让真实 SDK 对接伪造的 HTTP 传输层，对请求体和解码后的响应做断言。

---

## 编写会话存储

实现 `load` 和 `save`。`save` 每次收到的是完整历史，必须原子地替换已存内容。

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
        await self.redis.set(self.key, payload, ex=self.ttl)   # 单条 SET 是原子的
```

说明：

- `Message.to_dict()` / `Message.from_dict()` 是稳定的序列化格式，直接保存其输出即可。
- 每一步会调用两次 `save`，每次都带完整历史。对话很长且使用 SQL 存储时，可以用一张以 `(conversation_id, position)` 为键的只追加表，只插入超出已存数量的那些行，避免每次整体重写。
- 要防止同一对话上出现并发运行，可以在应用层给 `agent.run` 加锁，或在 `save` 中加入乐观版本号，冲突时抛出 `SessionError`。
- 会话里保存的是用户内容，访问控制、加密和保留期限应与其他用户数据一致。

---

## 编写上下文策略

实现 `prepare`。它接收完整的已存历史，返回一个 `ContextView`。

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

策略必须遵守的规则：

1. **只要视图与输入不同，就返回 `edited=True`。** Provider 依靠它来去掉那些只在历史未被改动时才有效的内容。
2. **保证视图合法。** 视图必须以一条"不是工具结果"的用户消息开头，每个 `ToolCall` 都要保留与之对应的 `ToolResult`。
3. **不要修改输入**，也不要修改其中的消息对象；需要改动时新建 `Message`。
4. **保持稳定。** 同样的历史应得到同样的视图，视图的前缀应尽量少变。每次调用都改变前缀的策略（比如上面这个示例，它的边界每轮都在移动）会让提示缓存失效；应当像 `SlidingWindow` 那样大步移动边界。

---

## 未包含的功能

以下功能是有意不做的，目的是让核心保持小而正确。它们都可以基于现有的扩展点实现。

| 未包含 | 如何添加 |
|---|---|
| 其他模型 API 的 Provider | 写一个 `Provider` 子类，约 60 行，见上文 |
| 结构化最终输出（带类型、经 Schema 校验的答案） | 在 Provider 上增加选项，使用 API 的结构化输出功能；或者加一个 "submit_answer" 工具，以其参数作为答案 |
| MCP 客户端 | 写一个函数列出 MCP 服务器的工具，把每个工具包装成 `Tool`，处理函数负责转发调用 |
| 内置工具（文件、Shell、网页） | 用普通的 `@tool` 函数实现。之所以不内置，是因为安全的实现取决于你的沙箱和权限设置 |
| 跨进程的持久化执行（用于长时间审批） | 会话加上 `agent.run(session=...)` 续跑已经覆盖了崩溃恢复；跨越数天的编排应该交给工作流引擎 |
| 消息中的图片和文件 | 在 `types.py` 增加新的内容块类型，并在 Provider 中实现编码 |
| 评测框架 | `ScriptedProvider` 覆盖确定性测试；针对真实模型的质量评测需要你自己的数据集和评分器 |
| 调用前的 token 计数 | 预算依据的是 Provider 在每次调用后上报的用量 |
