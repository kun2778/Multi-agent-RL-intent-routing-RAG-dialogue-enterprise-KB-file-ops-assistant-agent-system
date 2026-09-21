from typing import Dict, Any, Optional
from providers.registry import registry


class ModelRouter:
    """Route requests based on model field."""

    def __init__(self):
        self.routes = registry.routes
        self.default_provider = registry.default_provider

    def route(self, requested_model: str) -> tuple:
        """
        Route a model name to provider and upstream model.
        Returns: (provider_type, upstream_model)
        """
        route = self.routes.get(requested_model)
        if not route:
            if requested_model in registry.adapters:
                return requested_model, requested_model
            if self.default_provider:
                return self.default_provider, requested_model
            raise ValueError(f"Unknown model: {requested_model}")
        return route['provider'], route['upstream_model']

    def get_all_routes(self) -> Dict[str, Any]:
        return {alias: {"provider": prov, "upstream": model}
                for alias, prov in self.routes.items()
                for model in [registry.adapters.get(prov, {}).get('config', {}).get('models', [])]}