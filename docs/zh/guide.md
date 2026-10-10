# agentloom 使用指南

[English](../en/guide.md)

目录：[第一个 Agent](#1-第一个-agent) · [工具](#2-工具) · [运行](#3-运行-agent) · [会话](#4-会话) · [上限与重试](#5-上限与重试) · [人工审批](#6-人工审批) · [钩子](#7-钩子) · [上下文策略](#8-上下文策略) · [多 Agent](#9-多-agent) · [追踪](#10-追踪) · [测试](#11-测试你的-agent) · [Claude Provider](#12-claude-provider) · [常见问题](#13-常见问题)

> 说明：示例代码中的指令（instructions）和工具文档字符串使用英文书写，你可以换成中文，框架对语言没有要求。

---

## 1. 第一个 Agent

一个 Agent 由三部分组成：Provider（模型）、instructions（系统提示词）和工具。

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
print(result.output)   # 模型的最终文本
print(result.status)   # RunStatus.COMPLETED
print(result.usage)    # 所有步骤累计的 token 用量
```

`Agent` 对象只保存配置，不保存任何对话状态，所以同一个实例可以同时服务多个并发的运行。

---

## 2. 工具

### 定义工具

用 `@tool` 装饰一个函数。函数名、类型注解和文档字符串就是模型看到的全部信息，所以要把它们当作写给模型看的说明来写。

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

- `Args:` 之前的文字是工具描述。写清楚这个工具做什么、什么时候该用。
- `Args:` 下的每一项是对应参数的描述。
- 没有默认值的参数是必填参数。

### 支持的参数类型

| Python 注解 | JSON Schema |
|---|---|
| `str`、`int`、`float`、`bool` | `string`、`integer`、`number`、`boolean` |
| `list[T]`、`set[T]`、`tuple[T, ...]` | 元素为 `T` 的 `array` |
| `dict[str, T]` | 值为 `T` 的 `object` |
| `T \| None`、`Optional[T]`、`A \| B` | `anyOf` |
| `Literal["a", "b"]` | `enum` |
| `Enum` 子类 | 取成员值的 `enum`；函数收到的是枚举成员 |
| dataclass | `object`；函数收到的是实例 |
| pydantic v2 模型 | 模型自身的 Schema；函数收到的是校验后的实例 |
| `Annotated[T, "说明"]` | 带该说明的 `T` |

类型不受支持、缺少描述、使用了 `*args`/`**kwargs`、工具重名，这些问题会在**定义工具时**抛出 `ConfigurationError`，不会等到运行中途才暴露。

### 校验与错误

函数执行之前，参数会先按 Schema 校验。有问题时不会调用你的函数，而是把问题作为错误结果返回给模型，让它修正后重试：

```
Invalid arguments:
- $.limit: expected integer, got str
```

在工具内部，对于模型可以据此调整的失败，抛出 `ToolError`。错误信息会原样发给模型，所以要写成对它有指导意义的话：

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

其他任何异常同样会被捕获：带完整堆栈记录到名为 `agentloom` 的日志器，并以 `异常类型: 信息` 的形式反馈给模型。工具出错不会导致整次运行崩溃。

### 返回值

可以返回字符串，或任何可序列化为 JSON 的对象（字典、列表、数字、dataclass、pydantic 模型）。返回 `None` 时模型收到的是 `OK`。输出超过 `RunLimits.max_tool_output_chars`（默认 100,000 个字符）会被截断，同时告诉模型被截掉了多少字符，方便它改为请求更小范围的结果。

### 同步与异步

两种都支持。同步函数在工作线程中执行，慢查询不会阻塞事件循环。如果依赖的库有异步客户端，就用 `async def`。

### 工具选项

```python
@tool(requires_approval=True, parallel=False, timeout=30)
def refund(order_id: str, amount_cents: int) -> str:
    """Refund part of an order to the original payment method."""
    ...
```

| 选项 | 默认值 | 含义 |
|---|---|---|
| `name` | 函数名 | 模型看到的工具名 |
| `description` | 文档字符串摘要 | 覆盖文档字符串 |
| `requires_approval` | `False` | 每次调用都要经过 Agent 的审批函数（见[第 6 节](#6-人工审批)） |
| `parallel` | `True` | 有副作用的工具应设为 `False`。同一步里只要有一个不可并行的调用，这一步的所有调用就按顺序逐个执行 |
| `timeout` | `RunLimits.tool_timeout` | 超过这个秒数就放弃等待，并向模型报告超时 |

同步工具超时后，它所在的线程仍会继续执行，Python 无法强行终止线程。框架会告诉模型该操作可能已经生效。

### 上下文、依赖与单次运行内的状态

给函数加一个注解为 `ToolContext` 的参数，就能拿到运行时信息。这个参数对模型不可见。

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

- `ctx.deps` 就是调用 `run` 时传入的 `deps=`。适合放数据库连接和已认证的用户信息。**用户身份一定要从 `deps` 取，不要从模型提供的参数里取**：模型可能被诱导传入别人的 ID。
- `ctx.state` 是一个字典，同一次运行内的所有工具调用共享。
- `ctx.run_id`、`ctx.agent_name`、`ctx.call` 用于标识本次调用。

### 不用装饰器定义工具

当 Schema 来自别处（例如 OpenAPI 文档或 MCP 服务器）时，可以直接构造 `Tool`：

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

## 3. 运行 Agent

### 三种运行方式

```python
result = await agent.run("question")          # 异步，返回 RunResult
result = agent.run_sync("question")           # 阻塞式，适合脚本
async for event in agent.stream("question"):  # 异步，边运行边产出事件
    ...
```

在已经运行的事件循环里（比如 Jupyter 单元格或 FastAPI 处理函数中）调用 `run_sync` 会抛出 `ConfigurationError`，这种场合请用 `await agent.run(...)`。

### 运行结果

| 字段 | 含义 |
|---|---|
| `status` | 运行如何结束（见下表） |
| `ok` | 仅当 `status` 为 `COMPLETED` 时为 `True` |
| `output` | 本次运行最后一条助手消息的文本，可能为空 |
| `messages` | 完整历史，包含会话中此前的轮次 |
| `new_messages` | 本次运行新增的消息 |
| `usage` | token 总量，包含子 Agent 的用量 |
| `steps` | 模型调用次数 |
| `run_id` | 运行标识，每个事件上都带有 |

### 运行状态

可以预期的结束方式都以状态的形式返回，而不是抛异常，这样历史和已有的部分输出总是拿得到。

| `RunStatus` | 含义 | 常见处理 |
|---|---|---|
| `COMPLETED` | 模型完成了回答 | 使用 `output` |
| `MAX_STEPS` | 达到 `RunLimits.max_steps` | 展示已有进展；合适的话在同一会话上再发起一次运行继续 |
| `TOKEN_BUDGET` | 达到 `RunLimits.max_total_tokens` | 同上 |
| `TIMEOUT` | 达到 `RunLimits.max_seconds` | 同上 |
| `TRUNCATED` | 最终回答触及输出 token 上限 | 调大 `max_tokens`，或让模型接着写 |
| `REFUSED` | 模型拒绝了这个请求 | 告知用户；不要原样重试 |
| `CONTEXT_OVERFLOW` | 对话已超出模型的上下文窗口 | 使用[上下文策略](#8-上下文策略) |

**在把 `output` 当作完整答案使用之前，务必先检查 `result.ok`，或调用 `result.raise_for_status()`。**

### 会抛出哪些异常

| 异常 | 何时抛出 |
|---|---|
| `ProviderError` | 模型调用失败且重试次数已用尽，或错误本身不可重试（请求有误、认证失败）。失败之前已完成的步骤已经保存在会话里 |
| `ConfigurationError` | 用法错误：既没有输入、会话里也没有待处理的用户消息；在事件循环中调用 `run_sync`；工具定义有误 |
| `SessionError` | 会话存储读写失败 |
| 你的钩子或审批函数抛出的异常 | 原样向上传播 |

### 流式输出

```python
from agentloom import TextDelta, ToolCallStarted, ToolCallFinished, RetryScheduled, RunFinished

async for event in agent.stream("Plan a 3-day trip to Kyoto"):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStarted):
        print(f"\n[calling {event.call.name}]")
    elif isinstance(event, RetryScheduled):
        print("\n[retrying]")       # 失败的那次尝试已经输出的文字作废
    elif isinstance(event, RunFinished):
        result = event.result       # 总是最后一个事件
```

事件按以下顺序出现：先是 `RunStarted`；每一步依次可能有 `StepStarted`、`ThinkingDelta`\*、`TextDelta`\*、`RetryScheduled`\*、`ModelResponded`、`ToolCallStarted`\*、`ToolCallFinished`\*；最后是 `RunFinished`。每个事件都带 `run_id`。

---

## 4. 会话

不传会话时，每次运行都从零开始。传入会话就可以让对话延续下去：

```python
from agentloom import FileSession, InMemorySession

session = FileSession("data/sessions/user_42.json")   # 或 InMemorySession()

await agent.run("My name is Ada.", session=session)
result = await agent.run("What's my name?", session=session)
```

- `InMemorySession` 的生命周期与对象相同。
- `FileSession` 每个对话存成一个 JSON 文件。每次保存都是先写临时文件、刷盘、再原子重命名覆盖旧文件，因此进程崩溃不会留下写了一半的历史。

一个对话对应一个会话；不要让两个运行同时使用同一个会话：两边都会保存完整历史，后保存的会覆盖先保存的。

### 崩溃恢复

每次模型响应之后、每批工具结果之后，历史都会保存一次。如果进程恰好在这两个时间点之间退出，存下来的历史就会以"没有结果的工具调用"结尾。下次运行时框架会自动修复：为每个没有结果的调用补一条错误结果，告诉模型这次调用被打断了，可能已经生效，也可能没有。

想在不追加新用户消息的前提下继续一次被打断的运行，省略输入即可：

```python
result = await agent.run(session=session)
```

由于被打断的工具有可能已经执行过，**有副作用的工具应当做成幂等的**（接受幂等键，或执行前先检查当前状态）。框架能告诉模型"调用被打断了"，但无法撤销已经发生的操作。

如需把会话存进数据库，参见[编写会话存储](architecture.md#编写会话存储)。

---

## 5. 上限与重试

```python
from agentloom import Agent, RunLimits, RetryPolicy

agent = Agent(
    provider,
    tools=[...],
    limits=RunLimits(
        max_steps=25,                 # 每次运行的模型调用次数
        max_total_tokens=500_000,     # None 表示不限
        max_seconds=300,              # None 表示不限；在步与步之间检查
        max_repeated_calls=5,         # 每次运行内允许的完全相同的工具调用次数
        tool_timeout=120,             # 单次工具调用的默认超时（秒）
        max_parallel_tools=8,         # 同一步内并发执行的工具数量
        max_tool_output_chars=100_000,
    ),
    retry=RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0),
)
```

上面写的都是默认值（`max_total_tokens` 和 `max_seconds` 的默认值是 `None`）。

**重复调用。** 模型卡住时，常常会反复发出同一个调用。同一个工具以完全相同的参数在一次运行内被调用满 `max_repeated_calls` 次后，再出现相同的调用就不再执行，而是返回一条错误，提示模型换一种做法。对于本来就需要轮询的工具，可以调高这个上限。

**预算在步与步之间检查。** 已经发出的那一步会执行完，所以一次运行最多可能比 `max_total_tokens` 或 `max_seconds` 多出一步。

**重试。** 当 Provider 把错误标记为可重试时（限流、过载、5xx、超时、连接失败），模型调用会被重试。等待时间在 0 到 `base_delay * 2^(attempt-1)` 之间随机取值，上限为 `max_delay`；如果服务端给出了 `retry-after`，则以它为准。`max_attempts=1` 表示不重试。每次重试都会发出一个 `RetryScheduled` 事件。

---

## 6. 人工审批

对代价高或难以撤销的操作，可以要求人工确认：

```python
from agentloom import Agent, Approval, ToolCall, ToolContext, tool

@tool(requires_approval=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email from the user's account."""
    ...

async def approver(call: ToolCall, ctx: ToolContext) -> bool | Approval:
    if call.name == "send_email" and call.arguments["to"].endswith("@ourcompany.com"):
        return True                                   # 内部邮件：直接放行
    answer = await ask_the_user(f"Run {call.name} with {dict(call.arguments)}?")
    return Approval(answer.yes, reason=answer.comment)

agent = Agent(provider, tools=[send_email], approval=approver)
```

- 审批函数能看到真实的调用参数，返回 `True`、`False` 或 `Approval(approved, reason)`。
- 被拒绝时工具不会执行；模型会被告知调用未获批准，并附上你给出的理由。
- **如果工具要求审批而 Agent 没有配置审批函数，调用会被拒绝。** 没有审批函数绝不等于"放行"。
- 含有需审批调用的那一步会逐个执行调用，审批提示不会相互重叠。

审批函数是在运行过程中被等待的。如果审批可能要等几个小时，不要让运行一直挂着：让审批函数先以"已提交审核"之类的理由拒绝，自己记录下这次请求，等有了结论之后，再在同一个会话上发起新的运行。

---

## 7. 钩子

钩子可以观察一次运行，并在三个位置介入。继承 `Hooks`，重写需要的方法即可。

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

| 方法 | 用途 | 抛出异常时 |
|---|---|---|
| `on_event(event)` | 观察所有事件 | 记录日志后忽略：观察者出错不会导致运行失败 |
| `before_model(request)` | 返回修改后的 `ModelRequest`，或返回 `None` 表示不改 | 运行失败 |
| `before_tool(call, ctx)` | 抛出 `ToolDenied` 可拦截这次调用 | `ToolDenied` 只拦截该调用；其他异常导致运行失败 |
| `after_tool(call, result, ctx)` | 返回用于替换的 `ToolResult`，或返回 `None` | 运行失败 |

多个钩子按传入的顺序执行。无论提示词怎么写都必须成立的规则，请放在 `before_tool` 里：提示词只能引导模型，真正能保证规则的是代码。

---

## 8. 上下文策略

上下文策略决定每次模型调用能看到已存历史中的哪一部分。已存的历史本身永远不会被修改。

```python
from agentloom import Agent, SlidingWindow, Summarizing

Agent(provider)                                                   # KeepAll，默认
Agent(provider, context=SlidingWindow(max_messages=80, keep=40))
Agent(provider, context=Summarizing(cheap_provider, max_messages=80, keep=40))
```

| 策略 | 行为 | 适用场景 |
|---|---|---|
| `KeepAll` | 发送全部历史 | 对话装得进上下文窗口。以当前模型的窗口大小，大多数情况都属于这一类 |
| `SlidingWindow` | 超过 `max_messages` 后丢弃最早的消息，保留约 `keep` 条 | 早期轮次不重要 |
| `Summarizing` | 裁剪点相同，但被丢弃的部分替换为模型写的摘要 | 长任务，且早先的决定仍然重要 |

**没有明确需要时，请保持 `KeepAll`。** 历史只增不改，Provider 的提示缓存才能生效，而提示缓存通常是能拿到的最大一笔成本节省。

两种会裁剪的策略在设计上尽量降低了副作用：

- **成块裁剪。** 以 `max_messages=80, keep=40` 为例，视图每 40 条消息才裁剪一次，而不是每轮都往前挪一条。两次裁剪之间，发给模型的前缀保持不变，缓存可以继续命中。
- **裁剪点落在安全边界上。** 视图总是从一条真正的用户消息开始，不会切在工具调用和它的结果之间。找不到安全边界时就不裁剪。
- **视图被编辑后不再回传推理块。** 有些模型会把推理块与产生它的那段对话绑定签名，历史被改动后再回传会被拒绝。策略只要丢弃过内容，请求就会被标记为 `history_edited`，Claude Provider 据此在回放的历史轮次中去掉推理块，只保留文本和工具调用。

`Summarizing` 每次裁剪会多调用一次模型，摘要缓存在策略对象的内存里。因此长期存在的对话应当搭配长期存在的策略实例。摘要用一个小而快的模型就够了。

---

## 9. 多 Agent

可以把一个 Agent 变成另一个 Agent 能调用的工具：

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

- 子 Agent 从**空上下文**开始，只会收到 `task` 这一个字符串。这种隔离正是目的所在：它搜索和阅读产生的大量内容不会进入主 Agent 的上下文，回来的只有结论。
- 同一步里发起的多个委派会并行执行。
- 子 Agent 的 token 用量会计入主 Agent 的 `result.usage`。
- 子 Agent 没有正常完成时（达到步数上限、拒答等），主 Agent 会收到一条说明原因的错误结果，并附带已有的部分输出。
- `deps` 会透传给子 Agent 的工具。

当某个子任务会把主上下文塞满，或者多个子任务可以并行时，再考虑引入第二个 Agent。如果只是固定顺序的几个步骤，用单个 Agent，或者用普通代码依次调用几次 `agent.run`，更简单也更省钱。

---

## 10. 追踪

```python
from agentloom import ConsoleTracer, JsonlTracer

agent = Agent(provider, tools=[...], hooks=[ConsoleTracer(), JsonlTracer("runs.jsonl")])
```

`ConsoleTracer` 每个动作向 stderr 打印一行：

```
> run 18d8c15e (agent)
  step 1: tool_use [in 812 / out 64 / cached 0]
  -> order_status({"order_id": "A-1001"})
  ok order_status 0.01s: {"status": "shipped", "eta": "2026-10-14"}
  step 2: end_turn [in 901 / out 48 / cached 812]
= completed after 2 step(s), 2637 tokens
```

`JsonlTracer` 把除流式增量之外的每个事件追加写成一行带时间戳的 JSON，足以还原一次运行做了什么、每一步花了多少。文件里含有提示词和工具数据，请按敏感数据对待。

想把追踪数据发到别处（OpenTelemetry、数据库），写一个带 `on_event` 的 `Hooks` 子类即可；`agentloom.tracing.event_to_dict(event)` 可以把事件转成便于 JSON 序列化的字典。

框架自身的告警（重试、工具异常、钩子出错）输出到名为 `agentloom` 的标准 `logging` 日志器。

---

## 11. 测试你的 Agent

`ScriptedProvider` 代替模型，按脚本作答，每次模型调用消耗脚本中的一项。

```python
from agentloom import Agent, RunStatus, RunLimits
from agentloom.testing import ScriptedProvider, call

async def test_refund_requires_lookup_first():
    model = ScriptedProvider([
        call("get_order", order_id="A-1"),             # 第 1 步：模型调用一个工具
        [call("refund", order_id="A-1", amount_cents=500),
         call("notify", order_id="A-1")],              # 第 2 步：同时发起两个调用
        "Refunded $5.00.",                             # 第 3 步：最终回答
    ])
    agent = Agent(model, tools=[get_order, refund, notify])

    result = await agent.run("Refund $5 on A-1")

    assert result.status is RunStatus.COMPLETED
    sent_back = model.requests[1].messages[-1].tool_results   # 模型实际看到的工具结果
    assert not sent_back[0].is_error
```

脚本中的每一项可以是：字符串（最终回答）、一个 `ToolCall` 或其列表、一个 `ModelResponse`（用于控制停止原因或用量）、一个异常实例（会被抛出，用于测试重试），或者一个 `request -> 脚本项` 的函数。

`model.requests` 记录了每一次请求，可以据此断言 Agent 实际发出了什么。如果 Agent 调用模型的次数超过脚本的项数，Provider 会抛出异常，从而发现意料之外的循环。

这种方式可以确定性地测试你的工具、上限、钩子和审批逻辑。它测不了真实模型在你的提示词下会如何表现，那需要针对真实模型做评测。

---

## 12. Claude Provider

```python
from agentloom.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    "claude-opus-5-5",        # 默认模型
    max_tokens=64_000,        # 每次调用的默认输出上限
    effort=None,              # "low" | "medium" | "high" | "xhigh" | "max"；None 表示用模型默认值
    thinking=None,            # 例如 {"type": "adaptive", "display": "summarized"}
    cache=True,               # 自动提示缓存
    refusal_fallback=True,    # 见下文
    eager_tool_streaming=True,
    betas=(),
    extra=None,               # 其他任意 Messages API 参数
)
```

它替你处理了这些事：

- **始终使用流式请求**，长输出不会触发 HTTP 超时。
- **在后续轮次原样回传模型自己的内容块**（包括推理块），这是 API 的要求。
- **开启提示缓存**（顶层 `cache_control`）。同一会话内请保持 instructions 和工具列表不变，缓存才能命中；可以通过 `result.usage.cache_read_tokens` 确认。
- **创建客户端时关闭了 SDK 自带的重试**，因为重试由 Agent 的 `RetryPolicy` 负责，并会以事件形式上报。如果传入自己的客户端，请把它的 `max_retries` 设为 0，避免重复重试。

想通过 `ThinkingDelta` 事件看到推理过程，请传入 `thinking={"type": "adaptive", "display": "summarized"}`；当前模型默认返回的推理块文本是空的。

**拒答回退（refusal fallback）。** 部分当前的 Claude 模型带有安全分类器，可能拒绝某个请求。`refusal_fallback=True`（默认）时，Provider 会通过 beta 参数 `fallbacks="default"` 让 API 把被拒绝的请求交给回退模型重新处理。该选项只对 `claude-fable-5-1`、`claude-opus-5-5`、`claude-opus-5`、`claude-sonnet-5-5` 生效，并且会改走 SDK 的 beta 接口。**以下情况请设置 `refusal_fallback=False`**：使用 Amazon Bedrock、Vertex AI、Microsoft Foundry，或使用的代理不接受该参数，或者你不希望请求被其他模型处理。关闭后，拒答会让运行以 `RunStatus.REFUSED` 结束。

**其他平台。** 传入 Anthropic SDK 提供的平台客户端：

```python
from anthropic import AsyncAnthropicVertex

provider = AnthropicProvider(
    "claude-opus-5-5",
    client=AsyncAnthropicVertex(project_id="my-project", region="global", max_retries=0),
    refusal_fallback=False,
)
```

**查看实际请求。** `provider.build_params(request)` 返回将要发送的全部关键字参数，写测试和排查问题时很有用。

接入其他厂商的模型，参见[编写 Provider](architecture.md#编写-provider)。

---

## 13. 常见问题

| 现象 | 可能的原因与解决办法 |
|---|---|
| `ConfigurationError: AnthropicProvider needs the Anthropic SDK` | 执行 `pip install "agentloom[anthropic]"` |
| 创建 Provider 时 `ssl.create_default_context` 抛出 `FileNotFoundError` | 环境变量 `SSL_CERT_FILE` 指向了不存在的文件。取消该变量，或让它指向真实存在的 CA 证书文件 |
| `ProviderError: 401` | 缺少 API Key 或 Key 不正确 |
| `ProviderError: 400`，信息中提到 `fallbacks` 或未知的 beta | 你使用的端点不接受回退 beta：`AnthropicProvider(refusal_fallback=False)` |
| `ProviderError: 400`，信息中提到 `thinking` 块的 `signature` | 发给模型的历史被改动过（`before_model` 钩子、自定义上下文策略，或会话中途改了 instructions/工具）。保持历史只追加不修改，或者在请求上设置 `history_edited=True` 以去掉推理块 |
| 运行以 `MAX_STEPS` 结束 | 任务确实需要更多步（调大 `max_steps`），或者模型陷入了循环（查看追踪记录） |
| 模型反复收到 "Invalid arguments" | 工具描述或参数说明不够清楚。先改进文档字符串，再考虑别的办法 |
| `ConfigurationError: run_sync() was called from a running event loop` | 改用 `await agent.run(...)` |
| `cache_read_tokens` 一直是 0 | 前缀中有内容每次调用都在变：`instructions` 里带了时间戳、工具列表不固定，或者使用了会裁剪的上下文策略 |
