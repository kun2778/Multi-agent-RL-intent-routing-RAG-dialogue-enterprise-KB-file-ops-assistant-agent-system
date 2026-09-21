import os
from typing import Dict, Any, List
from providers.base import LLMRequest, LLMResponse

PROVIDER_MAP = {}

def get_adapter(provider_type: str, config: dict = None):
    from providers.adapters.openai_adapter import OpenAIAdapter
    from providers.adapters.anthropic_adapter import AnthropicAdapter
    from providers.adapters.deepseek_adapter import DeepSeekAdapter
    from providers.adapters.gemini_adapter import GeminiAdapter
    from providers.adapters.local_adapter import LocalAdapter
    ADAPTERS = {
        "openai": OpenAIAdapter, "anthropic": AnthropicAdapter,
        "deepseek": DeepSeekAdapter, "gemini": GeminiAdapter,
        "local": LocalAdapter,
    }
    adapter_class = ADAPTERS.get(provider_type)
    if not adapter_class:
        raise ValueError(f"Unknown provider type: {provider_type}")
    return adapter_class(config or {})