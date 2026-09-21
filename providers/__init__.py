# Providers package - lazy imports to avoid circular dependencies
from providers.registry import registry, ProviderRegistry
from providers.config_loader import ConfigLoader, get_config_loader
from providers.base import LLMProvider, LLMRequest, LLMResponse
from providers.secrets import SecretManager, get_api_key
from providers.adapters import get_adapter


class ProviderManager:
    """Unified manager for all LLM providers."""

    def __init__(self):
        self.config_loader = get_config_loader()
        self.secret_manager = SecretManager()
        self._setup()

    def _setup(self):
        from providers.adapters import get_adapter
        providers_config = self.config_loader.get_all_providers()
        for name, config in providers_config.items():
            provider_type = config.get('provider_type')
            try:
                adapter = get_adapter(provider_type, config)
                registry.register_adapter(name, adapter)
            except (ValueError, ImportError):
                continue
        routes_config = self.config_loader.load_models_config().get('routes', {})
        for alias, route in routes_config.items():
            registry.register_route(alias, route['provider'])
        registry.set_default_provider(self.config_loader.get_env('DEFAULT_PROVIDER', 'openai'))

    def get_adapter(self, provider_type: str):
        return registry.get_adapter(provider_type)

    def get_api_key(self, model: str = "openai") -> str:
        return get_api_key(model)

    def chat(self, model: str, messages: list, **kwargs) -> LLMResponse:
        provider_type = registry.get_route(model) or registry.default_provider
        adapter = registry.get_adapter(provider_type)
        api_key = self.get_api_key(provider_type)
        request = LLMRequest(model=model, messages=messages, api_key=api_key, **kwargs)
        return adapter.chat_completions(request)

    def list_available_models(self) -> dict:
        models = {}
        for name, adapter in registry.adapters.items():
            try:
                models[name] = adapter.list_models()
            except Exception:
                models[name] = []
        return models


__all__ = [
    "ProviderManager", "registry", "ProviderRegistry",
    "LLMProvider", "LLMRequest", "LLMResponse",
    "SecretManager", "get_api_key",
    "ConfigLoader", "get_config_loader", "get_adapter",
]