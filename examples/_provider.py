"""Choose the model for the examples from the environment.

# Claude (the default)
export ANTHROPIC_API_KEY=...

# Any OpenAI-compatible API; this wins when both are configured
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.deepseek.com MODEL=deepseek-chat
"""

import os

from agentloom import Provider


def provider_from_env() -> Provider:
    model = os.environ.get("MODEL")
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        if not model:
            raise SystemExit("Set MODEL to a model ID that your OpenAI-compatible endpoint serves.")
        from agentloom.providers.openai import OpenAIProvider

        return OpenAIProvider(model)

    from agentloom.providers.anthropic import AnthropicProvider

    return AnthropicProvider(model) if model else AnthropicProvider()
