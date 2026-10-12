"""OpenAIProvider against the real SDK, with the HTTP layer replaced by a local fake.

These tests check the request we build and how we read the SDK's stream. They
do not prove any particular service accepts the request; that needs a real key.
"""

import json

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx2")

from agentloom import (  # noqa: E402
    Agent,
    Message,
    ModelRequest,
    ProviderError,
    RetryPolicy,
    StopReason,
    TextBlock,
    TextChunk,
    ThinkingChunk,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    tool,
)
from agentloom.providers.openai import OpenAIProvider  # noqa: E402


def sse(*chunks: dict) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def chunk(delta: dict | None = None, finish: str | None = None, **top) -> dict:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": choices,
        **top,
    }


def tool_delta(index: int, arguments: str, id: str | None = None, name: str | None = None) -> dict:
    call: dict = {"index": index, "function": {"arguments": arguments}}
    if id is not None:
        call |= {"id": id, "type": "function"}
        call["function"]["name"] = name
    return {"tool_calls": [call]}


USAGE = {
    "prompt_tokens": 112,
    "completion_tokens": 34,
    "total_tokens": 146,
    "prompt_tokens_details": {"cached_tokens": 100},
}

TOOL_CALL_REPLY = sse(
    chunk({"role": "assistant", "content": ""}),
    chunk({"reasoning_content": "Need the weather."}),
    chunk({"content": "Let me "}),
    chunk({"content": "check."}),
    chunk(tool_delta(0, "", id="call_1", name="weather")),
    chunk(tool_delta(0, '{"city": ')),
    chunk(tool_delta(0, '"Oslo"}')),
    chunk({}, finish="tool_calls"),
    chunk(usage=USAGE),
)


def text_reply(text: str, finish: str = "stop") -> bytes:
    return sse(chunk({"content": text}), chunk({}, finish=finish), chunk(usage=USAGE))


class FakeApi:
    """Serves queued responses and records every request body."""

    def __init__(self, *responses: "httpx.Response | bytes") -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: "httpx.Request") -> "httpx.Response":
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, bytes):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "x-request-id": "req_1"},
                content=response,
            )
        return response

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)

    def provider(self, model: str = "test-model", **options) -> OpenAIProvider:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="https://llm.example/v1",
            max_retries=0,
            http_client=openai.DefaultAsyncHttpxClient(transport=httpx.MockTransport(self.handler)),
        )
        return OpenAIProvider(model, client=client, **options)


async def test_stream_maps_chunks_and_final_message():
    provider = FakeApi(TOOL_CALL_REPLY).provider()

    chunks = [c async for c in provider.stream(ModelRequest([Message.user("Weather?")]))]

    assert [c.text for c in chunks if isinstance(c, TextChunk)] == ["Let me ", "check."]
    assert [c.text for c in chunks if isinstance(c, ThinkingChunk)] == ["Need the weather."]
    response = chunks[-1].response
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.model == "test-model" and response.request_id == "req_1"
    # prompt_tokens includes the cached tokens; they are reported separately.
    assert response.usage == Usage(input_tokens=12, output_tokens=34, cache_read_tokens=100)
    message = response.message
    assert message.content == [
        TextBlock("Let me check."),
        ToolCall("call_1", "weather", {"city": "Oslo"}),
    ]
    assert message.provider == "openai"
    assert message.provider_data == {"reasoning_content": "Need the weather."}


async def test_parallel_tool_calls_are_assembled_by_index():
    reply = sse(
        chunk(tool_delta(0, "", id="call_a", name="weather")),
        chunk(tool_delta(1, "", id="call_b", name="weather")),
        chunk(tool_delta(1, '{"city": "Rome"}')),
        chunk(tool_delta(0, '{"city": "Oslo"}')),
        chunk({}, finish="stop"),  # some servers say "stop" even with tool calls
    )
    response = await FakeApi(reply).provider().complete(ModelRequest([Message.user("x")]))

    assert response.message.content == [
        ToolCall("call_a", "weather", {"city": "Oslo"}),
        ToolCall("call_b", "weather", {"city": "Rome"}),
    ]
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.usage == Usage()  # the server sent none


async def test_request_shape_defaults():
    api = FakeApi(text_reply("hi"))
    spec = ToolSpec("weather", "Get the weather.", {"type": "object", "properties": {}})

    await api.provider().complete(
        ModelRequest([Message.user("Weather?")], system="Be brief.", tools=[spec])
    )

    request = api.requests[0]
    body = api.body()
    assert request.url == "https://llm.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body == {
        "model": "test-model",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get the weather.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


async def test_request_shape_options():
    api = FakeApi(text_reply("hi"), text_reply("hi"))
    provider = api.provider(
        max_tokens=1000,
        max_tokens_param="max_completion_tokens",
        include_usage=False,
        extra={"temperature": 0},
    )

    await provider.complete(ModelRequest([Message.user("x")]))
    await provider.complete(ModelRequest([Message.user("x")], max_tokens=50))

    first, second = api.body(0), api.body(1)
    assert first["max_completion_tokens"] == 1000 and second["max_completion_tokens"] == 50
    assert first["temperature"] == 0
    assert "stream_options" not in first and "max_tokens" not in first and "tools" not in first


def history() -> list[Message]:
    return [
        Message.user("Weather?"),
        Message(
            "assistant",
            [TextBlock("Let me check."), ToolCall("call_1", "weather", {"city": "Oslo"})],
            provider="openai",
            provider_data={"reasoning_content": "Need the weather."},
        ),
        Message(
            "user",
            [
                TextBlock("and tomorrow?"),
                ToolResult("call_1", "rain"),
                ToolResult("call_2", "no such city", is_error=True),
            ],
        ),
        Message.assistant("Rain."),
    ]


def test_history_is_converted_to_chat_messages():
    provider = OpenAIProvider("m", client=object())

    messages = provider.build_params(ModelRequest(history()))["messages"]

    assert messages == [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Oslo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "rain"},
        {"role": "tool", "tool_call_id": "call_2", "content": "Error: no such city"},
        {"role": "user", "content": "and tomorrow?"},
        {"role": "assistant", "content": "Rain."},
    ]


def test_tool_call_without_text_has_null_content():
    provider = OpenAIProvider("m", client=object())
    foreign = Message("assistant", [ToolCall("c1", "weather", {})], provider="anthropic")

    messages = provider.build_params(ModelRequest([Message.user("x"), foreign]))["messages"]

    assert messages[1]["content"] is None and messages[1]["tool_calls"][0]["id"] == "c1"


def test_reasoning_is_replayed_only_when_asked_and_history_is_unedited():
    provider = OpenAIProvider("m", client=object(), replay_reasoning=True)

    def replayed(provider: OpenAIProvider, **request) -> bool:
        params = provider.build_params(ModelRequest(history(), **request))
        return "reasoning_content" in params["messages"][1]

    assert replayed(provider)
    assert not replayed(provider, history_edited=True)
    assert not replayed(OpenAIProvider("m", client=object()))


async def test_truncated_tool_arguments_are_reported_not_raised():
    reply = sse(
        chunk(tool_delta(0, '{"city": "Os', id="call_1", name="weather")),
        chunk({}, finish="length"),
    )
    response = await FakeApi(reply).provider().complete(ModelRequest([Message.user("x")]))

    assert response.stop_reason is StopReason.MAX_TOKENS
    assert response.message.content == [ToolCall("call_1", "weather", {})]


async def test_malformed_tool_arguments_are_retryable():
    reply = sse(
        chunk(tool_delta(0, '{"city": ', id="call_1", name="weather")),
        chunk({}, finish="tool_calls"),
    )
    with pytest.raises(ProviderError) as caught:
        await FakeApi(reply).provider().complete(ModelRequest([Message.user("x")]))
    assert caught.value.retryable and "weather" in str(caught.value)


async def test_stream_without_finish_reason_is_retryable():
    reply = sse(chunk({"content": "Let me"}))
    with pytest.raises(ProviderError) as caught:
        await FakeApi(reply).provider().complete(ModelRequest([Message.user("x")]))
    assert caught.value.retryable and caught.value.request_id == "req_1"


@pytest.mark.parametrize(
    "delta, finish, stop, text",
    [
        ({"content": "partial"}, "length", StopReason.MAX_TOKENS, "partial"),
        ({"content": ""}, "content_filter", StopReason.REFUSAL, ""),
        ({"refusal": "I can't help with that."}, "stop", StopReason.REFUSAL, "I can't help"),
    ],
)
async def test_stop_reasons(delta, finish, stop, text):
    reply = sse(chunk(delta), chunk({}, finish=finish))
    response = await FakeApi(reply).provider().complete(ModelRequest([Message.user("x")]))

    assert response.stop_reason is stop
    assert response.message.text.startswith(text)


def error_response(status: int, message: str, headers: dict | None = None) -> "httpx.Response":
    body = {"error": {"message": message, "type": "error", "code": None}}
    return httpx.Response(status, headers={"x-request-id": "req_9", **(headers or {})}, json=body)


@pytest.mark.parametrize(
    "response, retryable, retry_after",
    [
        (error_response(429, "rate limit happened", {"retry-after": "7"}), True, 7.0),
        (error_response(503, "overload happened"), True, None),
        (error_response(500, "server error happened"), True, None),
        (error_response(400, "bad request happened"), False, None),
        (error_response(401, "auth happened"), False, None),
        (error_response(404, "model not found happened"), False, None),
    ],
)
async def test_api_errors_are_classified(response, retryable, retry_after):
    provider = FakeApi(response).provider()

    with pytest.raises(ProviderError) as caught:
        await provider.complete(ModelRequest([Message.user("x")]))

    error = caught.value
    assert error.status_code == response.status_code
    assert error.retryable is retryable and error.retry_after == retry_after
    assert error.request_id == "req_9"
    assert "happened" in str(error)


async def test_connection_errors_are_retryable():
    def refuse(request):
        raise httpx.ConnectError("no route", request=request)

    client = openai.AsyncOpenAI(
        api_key="k",
        max_retries=0,
        http_client=openai.DefaultAsyncHttpxClient(transport=httpx.MockTransport(refuse)),
    )
    with pytest.raises(ProviderError) as caught:
        await OpenAIProvider("m", client=client).complete(ModelRequest([Message.user("x")]))
    assert caught.value.retryable and caught.value.status_code is None


async def test_full_agent_round_trip_through_the_sdk():
    @tool
    def weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Rain in {city}"

    api = FakeApi(
        error_response(503, "overloaded"),
        TOOL_CALL_REPLY,
        text_reply("Rain in Oslo."),
    )
    agent = Agent(api.provider(), tools=[weather], retry=RetryPolicy(base_delay=0))

    result = await agent.run("Weather in Oslo?")

    assert result.ok and result.output == "Rain in Oslo." and result.steps == 2
    assert len(api.requests) == 3  # one overloaded attempt, then two successful calls
    second_turn = api.body()["messages"]
    assert [m["role"] for m in second_turn] == ["user", "assistant", "tool"]
    assert second_turn[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Oslo"}'
    assert second_turn[2] == {"role": "tool", "tool_call_id": "call_1", "content": "Rain in Oslo"}
