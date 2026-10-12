"""Any OpenAI-compatible Chat Completions API (``pip install agentloom[openai]``).

Covers OpenAI itself and the many services and local servers that speak the
same protocol (DeepSeek, Qwen, Moonshot, OpenRouter, Ollama, vLLM, ...): pass
their ``base_url``.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ..errors import ConfigurationError, ProviderError
from ..types import (
    Block,
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    Usage,
)
from .base import Completed, Provider, ProviderChunk, TextChunk, ThinkingChunk

_STOP_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}
# Field names different servers use for streamed reasoning text.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


class OpenAIProvider(Provider):
    """Adapter for the Chat Completions API and services compatible with it.

    Args:
        model: Model ID as the endpoint names it (``"gpt-5"``, ``"deepseek-chat"``,
            ``"qwen-max"``, ``"llama3.3"``, ...).
        base_url: Endpoint root, for example ``"https://api.deepseek.com"`` or
            ``"http://localhost:11434/v1"``. ``None`` uses the SDK default
            (``OPENAI_BASE_URL``, else OpenAI).
        api_key: ``None`` reads ``OPENAI_API_KEY``. Local servers that need no
            key still require a non-empty placeholder.
        client: An ``openai.AsyncOpenAI`` (or ``AsyncAzureOpenAI``). When given,
            ``base_url`` and ``api_key`` are ignored. When omitted, one is
            created with SDK retries turned off because the agent's
            ``RetryPolicy`` handles retries.
        max_tokens: Default output cap per call. ``None`` sends no cap and
            leaves it to the endpoint.
        max_tokens_param: Request field that carries the cap. Most compatible
            servers expect ``"max_tokens"``; OpenAI's reasoning models require
            ``"max_completion_tokens"``.
        include_usage: Ask for token usage in the stream
            (``stream_options.include_usage``). Turn off for servers that
            reject the option; usage is then reported as zero unless the
            server sends it anyway.
        replay_reasoning: Send the model's ``reasoning_content`` back on later
            turns. Leave off unless your endpoint asks for it.
        extra: Additional request parameters merged in last
            (``temperature``, ``reasoning_effort``, ``extra_body``, ...).
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        max_tokens: int | None = None,
        max_tokens_param: str = "max_tokens",
        include_usage: bool = True,
        replay_reasoning: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ConfigurationError(
                "OpenAIProvider needs the OpenAI SDK: pip install 'agentloom[openai]'"
            ) from exc
        self._sdk: Any = openai
        self._owns_client = client is None
        self.client: Any = (
            client
            if client is not None
            else openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_tokens_param = max_tokens_param
        self.include_usage = include_usage
        self.replay_reasoning = replay_reasoning
        self.extra = dict(extra or {})

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()

    # --- request -----------------------------------------------------------

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        """The keyword arguments for ``chat.completions.create``. Public for inspection."""
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            if message.role == "assistant":
                messages.append(
                    self._assistant_param(message, strip_reasoning=request.history_edited)
                )
                continue
            # Tool results must directly follow the assistant turn that asked for them.
            for result in message.tool_results:
                # The protocol has no error flag, so the text has to say it.
                content = f"Error: {result.content}" if result.is_error else result.content
                messages.append(
                    {"role": "tool", "tool_call_id": result.tool_call_id, "content": content}
                )
            if message.text:
                messages.append({"role": "user", "content": message.text})

        params: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
        if self.include_usage:
            params["stream_options"] = {"include_usage": True}
        max_tokens = request.max_tokens or self.max_tokens
        if max_tokens is not None:
            params[self.max_tokens_param] = max_tokens
        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": dict(spec.input_schema),
                    },
                }
                for spec in request.tools
            ]
        params.update(self.extra)
        return params

    def _assistant_param(self, message: Message, *, strip_reasoning: bool) -> dict[str, Any]:
        calls = message.tool_calls
        param: dict[str, Any] = {"role": "assistant", "content": message.text or None}
        if calls:
            param["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                }
                for call in calls
            ]
        else:
            param["content"] = message.text
        if (
            self.replay_reasoning
            and not strip_reasoning
            and message.provider == self.name
            and isinstance(message.provider_data, dict)
            and message.provider_data.get("reasoning_content")
        ):
            param["reasoning_content"] = message.provider_data["reasoning_content"]
        return param

    # --- call --------------------------------------------------------------

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderChunk]:
        sdk = self._sdk
        params = self.build_params(request)
        text: list[str] = []
        reasoning: list[str] = []
        refusal: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish: str | None = None
        usage: Any = None
        model = ""
        request_id: str | None = None
        try:
            async with await self.client.chat.completions.create(**params) as stream:
                with contextlib.suppress(AttributeError):
                    request_id = stream.response.headers.get("x-request-id")
                async for chunk in stream:
                    model = getattr(chunk, "model", None) or model
                    usage = getattr(chunk, "usage", None) or usage
                    for choice in chunk.choices or ():
                        finish = choice.finish_reason or finish
                        delta = choice.delta
                        if delta is None:
                            continue
                        for field in _REASONING_FIELDS:
                            thought = getattr(delta, field, None)
                            if isinstance(thought, str) and thought:
                                reasoning.append(thought)
                                yield ThinkingChunk(thought)
                                break
                        if delta.content:
                            text.append(delta.content)
                            yield TextChunk(delta.content)
                        if getattr(delta, "refusal", None):
                            refusal.append(delta.refusal)
                        for part in delta.tool_calls or ():
                            call = calls.setdefault(
                                getattr(part, "index", None) or 0,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            call["id"] = part.id or call["id"]
                            function = part.function
                            if function is not None:
                                call["name"] = function.name or call["name"]
                                call["arguments"] += function.arguments or ""
        except sdk.RateLimitError as exc:
            raise self._error(exc, retryable=True) from exc
        except sdk.APIStatusError as exc:
            raise self._error(
                exc, retryable=exc.status_code in (408, 409) or exc.status_code >= 500
            ) from exc
        except sdk.APIConnectionError as exc:  # includes timeouts
            raise ProviderError(f"connection error: {exc}", retryable=True) from exc
        except sdk.APIError as exc:  # an error event inside an already-open stream
            raise ProviderError(f"stream error: {exc}", retryable=True) from exc

        if finish is None:
            # Without a finish reason there is no telling a complete answer from a cut-off one.
            raise ProviderError(
                f"{self.name}: stream ended without a finish reason",
                retryable=True,
                request_id=request_id,
            )

        content: list[Block] = []
        if text:
            content.append(TextBlock("".join(text)))
        for index in sorted(calls):
            content.append(self._tool_call(calls[index], truncated=finish == "length"))
        stop = _STOP_REASONS.get(finish, StopReason.OTHER)
        if calls and stop in (StopReason.END_TURN, StopReason.OTHER):
            stop = StopReason.TOOL_USE  # some servers report "stop" alongside tool calls
        if refusal and not content:
            content.append(TextBlock("".join(refusal)))
            stop = StopReason.REFUSAL

        thoughts = "".join(reasoning)
        yield Completed(
            ModelResponse(
                message=Message(
                    role="assistant",
                    content=content,
                    provider=self.name,
                    provider_data={"reasoning_content": thoughts} if thoughts else None,
                ),
                stop_reason=stop,
                usage=self._usage(usage),
                model=model or self.model,
                request_id=request_id,
            )
        )

    @staticmethod
    def _tool_call(raw: Mapping[str, str], *, truncated: bool) -> ToolCall:
        arguments: Any = {}
        if raw["arguments"].strip():
            try:
                arguments = json.loads(raw["arguments"])
            except ValueError as exc:
                # A truncated turn is reported as MAX_TOKENS and the agent refuses
                # to run its calls; anything else is a garbled turn worth re-issuing.
                if not truncated:
                    raise ProviderError(
                        f"malformed arguments for tool {raw['name']!r}: {exc}", retryable=True
                    ) from exc
        return ToolCall(
            raw["id"] or f"call_{uuid.uuid4().hex[:24]}",
            raw["name"],
            arguments if isinstance(arguments, dict) else {},
        )

    @staticmethod
    def _usage(usage: Any) -> Usage:
        if usage is None:
            return Usage()
        prompt = getattr(usage, "prompt_tokens", None) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) or 0
        return Usage(
            # prompt_tokens counts cached tokens too; Usage keeps them apart.
            input_tokens=max(prompt - cached, 0),
            output_tokens=getattr(usage, "completion_tokens", None) or 0,
            cache_read_tokens=cached,
        )

    @staticmethod
    def _error(exc: Any, *, retryable: bool) -> ProviderError:
        retry_after: float | None = None
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            retry_after = float(exc.response.headers.get("retry-after"))
        return ProviderError(
            f"{exc.status_code}: {getattr(exc, 'message', exc)}",
            retryable=retryable,
            status_code=exc.status_code,
            retry_after=retry_after,
            request_id=getattr(exc, "request_id", None),
        )
