import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ConfigLoader:
    """Load configuration from YAML files and environment variables."""

    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), '..', 'config')
        self._cache: Dict[str, Any] = {}

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        filepath = os.path.join(self.config_dir, filename)
        if filepath in self._cache:
            return self._cache[filepath]

        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        self._cache[filepath] = config
        return config

    def load_models_config(self) -> Dict[str, Any]:
        return self.load_yaml('models.yaml')

    def load_app_config(self) -> Dict[str, Any]:
        return self.load_yaml('app.yaml')

    def load_secrets_config(self) -> Dict[str, Any]:
        return self.load_yaml('secrets.yaml')

    def load_gateway_config(self) -> Dict[str, Any]:
        return self.load_yaml('gateway.yaml')

    def get_env(self, key: str, default: str = None) -> str:
        """Get value from environment variable."""
        return os.environ.get(key, default)

    def get_secret(self, key: str) -> str:
        """Get a secret value with fallback chain: env -> .env -> config."""
        value = self.get_env(key)
        if value:
            return value

        # Fallback to .env file
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if os.path.exists(env_path):
            from dotenv import load_dotenv
            load_dotenv(env_path)
            return os.environ.get(key)
        return None

    def get_all_providers(self) -> Dict[str, Any]:
        """Get all provider configurations."""
        models_config = self.load_models_config()
        return models_config.get('providers', {})

    def get_provider(self, name: str) -> Dict[str, Any]:
        """Get a specific provider configuration."""
        providers = self.get_all_providers()
        return providers.get(name, {})

    def get_model_route(self, alias: str) -> Dict[str, Any]:
        """Get model route by alias."""
        models_config = self.load_models_config()
        routes = models_config.get('routes', {})
        return routes.get(alias, {})


_config_loader: Optional[ConfigLoader] = None

def get_config_loader(config_dir: str = None) -> ConfigLoader:
    """Get singleton ConfigLoader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader
