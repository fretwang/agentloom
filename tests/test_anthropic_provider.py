"""AnthropicProvider against the real SDK, with the HTTP layer replaced by a local fake.

These tests check the request we build and how we read the SDK's stream. They
do not prove the live API accepts the request; that needs a real key.
"""

import json

import pytest

anthropic = pytest.importorskip("anthropic")
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
from agentloom.providers.anthropic import AnthropicProvider  # noqa: E402


def sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message_events(blocks: list[dict], stop_reason: str, model: str = "claude-opus-5-5") -> bytes:
    """Encode a complete Messages API streaming response for the given content blocks."""
    events: list[dict] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 7,
                },
            },
        }
    ]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": part} for part in block["parts"]]
        elif block["type"] == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": p} for p in block["json_parts"]]
        events.append({"type": "content_block_start", "index": index, "content_block": start})
        events += [{"type": "content_block_delta", "index": index, "delta": d} for d in deltas]
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 34},
        }
    )
    events.append({"type": "message_stop"})
    return sse(*events)


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
                200, headers={"content-type": "text/event-stream"}, content=response
            )
        return response

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)

    def provider(self, **options) -> AnthropicProvider:
        client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(
                transport=httpx.MockTransport(self.handler)
            ),
        )
        return AnthropicProvider(client=client, **options)


TOOL_CALL_REPLY = [
    {"type": "thinking", "thinking": "Need the weather.", "signature": "sig-abc"},
    {"type": "text", "parts": ["Let me ", "check."]},
    {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "weather",
        "json_parts": ['{"city": ', '"Oslo"}'],
    },
]


async def test_stream_maps_sdk_events_and_final_message():
    api = FakeApi(message_events(TOOL_CALL_REPLY, "tool_use"))
    provider = api.provider()

    chunks = [chunk async for chunk in provider.stream(ModelRequest([Message.user("Weather?")]))]

    assert [c.text for c in chunks if isinstance(c, TextChunk)] == ["Let me ", "check."]
    assert [c.text for c in chunks if isinstance(c, ThinkingChunk)] == ["Need the weather."]
    response = chunks[-1].response
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.model == "claude-opus-5-5"
    assert response.usage == Usage(
        input_tokens=12, output_tokens=34, cache_read_tokens=100, cache_write_tokens=7
    )
    message = response.message
    assert message.content == [
        TextBlock("Let me check."),
        ToolCall("toolu_1", "weather", {"city": "Oslo"}),
    ]
    assert message.provider == "anthropic"
    assert [b["type"] for b in message.provider_data] == ["thinking", "text", "tool_use"]
    assert message.provider_data[0]["signature"] == "sig-abc"
    json.dumps(message.provider_data)  # must be persistable by a session


async def test_request_shape_defaults():
    api = FakeApi(message_events([{"type": "text", "parts": ["hi"]}], "end_turn"))
    provider = api.provider()
    spec = ToolSpec("weather", "Get the weather.", {"type": "object", "properties": {}})

    await provider.complete(
        ModelRequest([Message.user("Weather?")], system="Be brief.", tools=[spec])
    )

    request = api.requests[0]
    body = api.body()
    assert request.url.path == "/v1/messages"
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 64_000
    assert body["stream"] is True
    assert body["system"] == "Be brief."
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["tools"] == [
        {
            "name": "weather",
            "description": "Get the weather.",
            "input_schema": {"type": "object", "properties": {}},
            "eager_input_streaming": True,
        }
    ]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Weather?"}]}]
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    assert "thinking" not in body and "output_config" not in body and "temperature" not in body


async def test_request_shape_options():
    api = FakeApi(message_events([{"type": "text", "parts": ["hi"]}], "end_turn"))
    provider = api.provider(
        model="claude-haiku-5-5",
        max_tokens=1000,
        effort="low",
        thinking={"type": "adaptive", "display": "summarized"},
        cache=False,
        eager_tool_streaming=False,
        extra={"metadata": {"user_id": "u1"}},
    )
    spec = ToolSpec("t", "A tool.", {"type": "object", "properties": {}})

    await provider.complete(ModelRequest([Message.user("x")], tools=[spec], max_tokens=50))

    body = api.body()
    assert body["model"] == "claude-haiku-5-5" and body["max_tokens"] == 50
    assert body["output_config"] == {"effort": "low"}
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["metadata"] == {"user_id": "u1"}
    assert "eager_input_streaming" not in body["tools"][0]
    # No fallback model exists for this model, so the request stays on the stable endpoint.
    assert "cache_control" not in body and "fallbacks" not in body
    assert "anthropic-beta" not in api.requests[0].headers


def history_with_reasoning() -> list[Message]:
    native = [
        {"type": "thinking", "thinking": "", "signature": "sig-abc"},
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Oslo"}},
    ]
    return [
        Message.user("Weather?"),
        Message(
            "assistant",
            [TextBlock("Let me check."), ToolCall("toolu_1", "weather", {"city": "Oslo"})],
            provider="anthropic",
            provider_data=native,
        ),
        Message("user", [TextBlock("and tomorrow?"), ToolResult("toolu_1", "rain", is_error=True)]),
    ]


def test_history_replays_native_blocks_unchanged():
    provider = AnthropicProvider(client=object())
    history = history_with_reasoning()

    params = provider.build_params(ModelRequest(history))

    assert params["messages"][1] == {"role": "assistant", "content": history[1].provider_data}
    assert params["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "rain", "is_error": True},
        {"type": "text", "text": "and tomorrow?"},
    ]


def test_edited_history_drops_reasoning_blocks():
    provider = AnthropicProvider(client=object())

    params = provider.build_params(ModelRequest(history_with_reasoning(), history_edited=True))

    assert [b["type"] for b in params["messages"][1]["content"]] == ["text", "tool_use"]


def test_messages_from_another_provider_use_the_neutral_view():
    provider = AnthropicProvider(client=object())
    foreign = Message(
        "assistant",
        [TextBlock(""), ToolCall("c1", "weather", {"city": "Oslo"})],
        provider="other",
        provider_data={"anything": 1},
    )

    params = provider.build_params(ModelRequest([Message.user("x"), foreign]))

    assert params["messages"][1]["content"] == [
        {"type": "tool_use", "id": "c1", "name": "weather", "input": {"city": "Oslo"}}
    ]


def error_response(status: int, kind: str, headers: dict | None = None) -> "httpx.Response":
    body = {"type": "error", "error": {"type": kind, "message": f"{kind} happened"}}
    return httpx.Response(status, headers={"request-id": "req_9", **(headers or {})}, json=body)


@pytest.mark.parametrize(
    "response, retryable, retry_after",
    [
        (error_response(429, "rate_limit_error", {"retry-after": "7"}), True, 7.0),
        (error_response(529, "overloaded_error"), True, None),
        (error_response(500, "api_error"), True, None),
        (error_response(400, "invalid_request_error"), False, None),
        (error_response(401, "authentication_error"), False, None),
        (error_response(404, "not_found_error"), False, None),
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

    client = anthropic.AsyncAnthropic(
        api_key="k",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx.MockTransport(refuse)),
    )
    with pytest.raises(ProviderError) as caught:
        await AnthropicProvider(client=client).complete(ModelRequest([Message.user("x")]))
    assert caught.value.retryable and caught.value.status_code is None


async def test_full_agent_round_trip_through_the_sdk():
    @tool
    def weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Rain in {city}"

    api = FakeApi(
        error_response(529, "overloaded_error"),
        message_events(TOOL_CALL_REPLY, "tool_use"),
        message_events([{"type": "text", "parts": ["Rain in Oslo."]}], "end_turn"),
    )
    agent = Agent(api.provider(), tools=[weather], retry=RetryPolicy(base_delay=0))

    result = await agent.run("Weather in Oslo?")

    assert result.ok and result.output == "Rain in Oslo." and result.steps == 2
    assert len(api.requests) == 3  # one overloaded attempt, then two successful calls
    second_turn = api.body()["messages"]
    assert [b["type"] for b in second_turn[1]["content"]] == ["thinking", "text", "tool_use"]
    assert second_turn[1]["content"][0]["signature"] == "sig-abc"
    assert second_turn[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Rain in Oslo"}],
    }
