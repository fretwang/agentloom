"""Claude via the official Anthropic SDK (``pip install agentloom[anthropic]``)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ..errors import ConfigurationError, ProviderError
from ..types import (
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from .base import Completed, Provider, ProviderChunk, TextChunk, ThinkingChunk

DEFAULT_MODEL = "claude-opus-5-5"

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models whose safety classifiers can decline a request and that accept
# `fallbacks: "default"` on the Claude API.
_FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
_REASONING_BLOCKS = frozenset({"thinking", "redacted_thinking"})
_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "pause_turn": StopReason.PAUSE,
    "refusal": StopReason.REFUSAL,
    "model_context_window_exceeded": StopReason.CONTEXT_OVERFLOW,
}


class AnthropicProvider(Provider):
    """Adapter for the Claude Messages API.

    Args:
        model: Model ID.
        client: An ``anthropic.AsyncAnthropic`` (or a platform client such as
            ``AsyncAnthropicVertex``). When omitted, one is created that reads
            credentials from the environment, with SDK retries turned off
            because the agent's ``RetryPolicy`` handles retries.
        max_tokens: Default output cap per call; requests are streamed, so a
            large value does not risk an HTTP timeout.
        effort: ``output_config.effort`` (``"low"`` .. ``"max"``). ``None``
            uses the model's default.
        thinking: Passed through as the ``thinking`` parameter. ``None`` omits
            it, which runs adaptive thinking on current models.
        cache: Enable automatic prompt caching (top-level ``cache_control``).
        refusal_fallback: On models that can decline a request, ask the API to
            re-run declined requests on a fallback model (beta). Turn off for
            Bedrock, Vertex AI, Foundry or proxies that reject the parameter.
        eager_tool_streaming: Stream tool inputs as they are generated. The
            agent validates every tool input itself, as this mode requires.
        betas: Extra ``anthropic-beta`` values. Any beta switches the call to
            ``client.beta.messages``.
        extra: Additional request parameters merged in last.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: Any = None,
        max_tokens: int = 64_000,
        effort: str | None = None,
        thinking: Mapping[str, Any] | None = None,
        cache: bool = True,
        refusal_fallback: bool = True,
        eager_tool_streaming: bool = True,
        betas: Sequence[str] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ConfigurationError(
                "AnthropicProvider needs the Anthropic SDK: pip install 'agentloom[anthropic]'"
            ) from exc
        self._sdk: Any = anthropic
        self._owns_client = client is None
        self.client: Any = client if client is not None else anthropic.AsyncAnthropic(max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = dict(thinking) if thinking is not None else None
        self.cache = cache
        self.refusal_fallback = refusal_fallback and model in _FALLBACK_MODELS
        self.eager_tool_streaming = eager_tool_streaming
        self.betas = list(betas)
        self.extra = dict(extra or {})

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()

    # --- request -----------------------------------------------------------

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        """The keyword arguments for ``messages.stream``. Public so it can be inspected."""
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": [
                self._message_param(m, strip_reasoning=request.history_edited)
                for m in request.messages
            ],
        }
        if request.system:
            params["system"] = request.system
        if request.tools:
            params["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": dict(spec.input_schema),
                    **({"eager_input_streaming": True} if self.eager_tool_streaming else {}),
                }
                for spec in request.tools
            ]
        if self.effort is not None:
            params["output_config"] = {"effort": self.effort}
        if self.thinking is not None:
            params["thinking"] = self.thinking
        if self.cache:
            params["cache_control"] = {"type": "ephemeral"}
        betas = list(self.betas)
        if self.refusal_fallback:
            params["fallbacks"] = "default"
            betas.append(_FALLBACK_BETA)
        if betas:
            params["betas"] = betas
        params.update(self.extra)
        return params

    def _message_param(self, message: Message, *, strip_reasoning: bool) -> dict[str, Any]:
        if (
            message.role == "assistant"
            and message.provider == self.name
            and isinstance(message.provider_data, list)
        ):
            # Replay the model's own blocks unchanged. Reasoning blocks are only
            # valid while everything before them is unchanged, so once a context
            # strategy has edited the history they must be left out.
            blocks = [
                block
                for block in message.provider_data
                if not (strip_reasoning and block.get("type") in _REASONING_BLOCKS)
            ]
            if blocks:
                return {"role": "assistant", "content": blocks}

        content: list[dict[str, Any]] = []
        # Tool results must precede any text in the same user turn.
        for block in sorted(message.content, key=lambda b: not isinstance(b, ToolResult)):
            if isinstance(block, TextBlock):
                if block.text:
                    content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCall):
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.arguments),
                    }
                )
            else:
                item: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.tool_call_id,
                    "content": block.content,
                }
                if block.is_error:
                    item["is_error"] = True
                content.append(item)
        return {"role": message.role, "content": content}

    # --- call --------------------------------------------------------------

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderChunk]:
        sdk = self._sdk
        params = self.build_params(request)
        messages_api = self.client.beta.messages if "betas" in params else self.client.messages
        try:
            async with messages_api.stream(**params) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextChunk(event.text)
                    elif (
                        event.type == "content_block_delta" and event.delta.type == "thinking_delta"
                    ):
                        yield ThinkingChunk(event.delta.thinking)
                final = await stream.get_final_message()
        except ValueError as exc:
            # The SDK could not parse a streamed tool input. No tool_use block
            # completed, so there is nothing to answer; re-issue the turn.
            raise ProviderError(f"malformed tool input in stream: {exc}", retryable=True) from exc
        except sdk.RateLimitError as exc:
            raise self._error(exc, retryable=True) from exc
        except sdk.APIStatusError as exc:
            raise self._error(
                exc, retryable=exc.status_code in (408, 409) or exc.status_code >= 500
            ) from exc
        except sdk.APIConnectionError as exc:  # includes timeouts
            raise ProviderError(f"connection error: {exc}", retryable=True) from exc
        yield Completed(self._response(final))

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

    def _response(self, final: Any) -> ModelResponse:
        content: list[TextBlock | ToolCall | ToolResult] = []
        for block in final.content:
            if block.type == "text":
                content.append(TextBlock(block.text))
            elif block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                content.append(ToolCall(block.id, block.name, arguments))
        usage = final.usage
        return ModelResponse(
            message=Message(
                role="assistant",
                content=list(content),
                provider=self.name,
                provider_data=final.to_dict()["content"],
            ),
            stop_reason=_STOP_REASONS.get(final.stop_reason or "", StopReason.OTHER),
            usage=Usage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
            ),
            model=final.model,
            request_id=getattr(final, "_request_id", None),
        )
