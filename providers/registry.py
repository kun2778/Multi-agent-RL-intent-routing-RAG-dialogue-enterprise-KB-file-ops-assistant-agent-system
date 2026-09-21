from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class ProviderRegistry:
    """Dynamic registry for LLM providers with factory pattern."""

    def __init__(self):
        self._providers: Dict[str, Any] = {}
        self._adapters: Dict[str, Any] = {}
        self._routes: Dict[str, str] = {}
        self._default_provider: str = "openai"

    def register_provider(self, name: str, provider_class, config: Dict[str, Any] = None):
        """Register a provider class with configuration."""
        self._providers[name] = {
            'class': provider_class,
            'config': config or {}
        }

    def register_adapter(self, provider_type: str, adapter_instance):
        """Register an adapter for a specific provider type."""
        self._adapters[provider_type] = adapter_instance

    def register_route(self, alias: str, provider_type: str):
        """Register a model alias route."""
        self._routes[alias] = provider_type

    def get_provider(self, name: str):
        """Get a provider instance by name."""
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")
        provider_info = self._providers[name]
        return provider_info['class'](**provider_info['config'])

    def get_adapter(self, provider_type: str):
        """Get an adapter by provider type."""
        if provider_type not in self._adapters:
            raise ValueError(f"Adapter for '{provider_type}' not registered")
        return self._adapters[provider_type]

    def get_route(self, alias: str) -> Optional[str]:
        """Get provider type for a model alias."""
        return self._routes.get(alias)

    def get_all_routes(self) -> Dict[str, str]:
        """Get all registered routes."""
        return dict(self._routes)

    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy provider names."""
        healthy = []
        for name, info in self._providers.items():
            try:
                instance = info['class'](**info['config'])
                if instance.health_check():
                    healthy.append(name)
            except Exception:
                continue
        return healthy

    def set_default_provider(self, name: str):
        """Set the default provider."""
        self._default_provider = name

    @property
    def providers(self) -> Dict[str, Any]:
        return dict(self._providers)

    @property
    def adapters(self) -> Dict[str, Any]:
        return dict(self._adapters)

    @property
    def routes(self) -> Dict[str, str]:
        return dict(self._routes)


# Global registry instance
registry = ProviderRegistry()


def setup_default_providers(config_loader=None):
    """Set up default providers from configuration."""
    from providers.adapters import (
        OpenAIAdapter, AnthropicAdapter, DeepSeekAdapter,
        GeminiAdapter, LocalAdapter
    )

    if config_loader is None:
        config_loader = get_config_loader()

    providers_config = config_loader.get_all_providers()

    for name, config in providers_config.items():
        registry.register_provider(name, config['adapter_class'], config)
        adapter_class = config.get('adapter_class')
        if adapter_class:
            registry.register_adapter(name, adapter_class(config))

    # Register routes
    routes_config = config_loader.load_models_config().get('routes', {})
    for alias, route in routes_config.items():
        registry.register_route(alias, route['provider'])

    registry.set_default_provider(config_loader.get_env('DEFAULT_PROVIDER', 'openai'))
