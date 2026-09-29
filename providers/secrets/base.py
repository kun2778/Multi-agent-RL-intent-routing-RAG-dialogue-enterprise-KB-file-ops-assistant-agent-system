import os
import yaml
import warnings
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


class SecretProvider(ABC):
    """Abstract base class for secret providers."""

    @abstractmethod
    def get_secret(self, key: str) -> str:
        """Retrieve a secret by key."""
        pass

    @abstractmethod
    def store_secret(self, key: str, value: str, ttl: int = None) -> bool:
        """Store a secret with optional TTL."""
        pass

    @abstractmethod
    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        pass

    @abstractmethod
    def list_secrets(self, prefix: str = "") -> List[str]:
        """List secrets with optional prefix."""
        pass


class EnvSecretProvider(SecretProvider):
    """Read secrets from environment variables and .env files."""

    def __init__(self, env_file: str = None):
        self.env_file = env_file
        if env_file and os.path.exists(env_file):
            from dotenv import load_dotenv
            load_dotenv(env_file)

    def get_secret(self, key: str) -> str:
        return os.environ.get(key)

    def store_secret(self, key: str, value: str, ttl: int = None) -> bool:
        os.environ[key] = value
        return True

    def delete_secret(self, key: str) -> bool:
        if key in os.environ:
            del os.environ[key]
        return True

    def list_secrets(self, prefix: str = "") -> List[str]:
        return [k for k in os.environ.keys() if k.startswith(prefix)]


class VaultSecretProvider(SecretProvider):
    """HashiCorp Vault client for dynamic secrets."""

    def __init__(self, vault_url: str, token: str = None, namespace: str = None,
                 engine: str = 'openai', timeout: float = 5.0):
        if not vault_url:
            raise ValueError("vault_url is required for VaultSecretProvider")
        self.vault_url = vault_url.rstrip('/')
        self.namespace = namespace
        self.token = token or os.environ.get('VAULT_TOKEN')
        self._client = None
        self._engine_path = engine or 'openai'
        self._timeout = timeout
        self._unavailable = False

    @property
    def available(self) -> bool:
        """False once the Vault endpoint failed, so the chain skips it."""
        return not self._unavailable

    def _get_client(self):
        """Get or create a Vault HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                base_url=self.vault_url,
                headers={
                    "X-Vault-Token": self.token or '',
                    "Content-Type": "application/json"
                },
                timeout=self._timeout
            )
            if self.namespace:
                self._client.headers["X-Vault-Namespace"] = self.namespace
        return self._client

    def _request(self, method: str, path: str, **kwargs):
        """Perform a request, returning None and opening the circuit on failure."""
        if self._unavailable:
            return None
        try:
            return getattr(self._get_client(), method)(path, **kwargs)
        except Exception:
            self._unavailable = True
            return None

    def enable_engine(self) -> bool:
        """Enable the OpenAI secrets engine in Vault."""
        resp = self._request("post", f"/v1/sys/plugins/catalog/{self._engine_path}", json={
            "type": "secret"
        })
        return resp is not None and resp.status_code in (200, 204)

    def configure_engine(self, api_key: str, organization_id: str, rotation_period: str = "720h") -> bool:
        """Configure the OpenAI secrets engine."""
        resp = self._request("post", f"/v1/{self._engine_path}/config", json={
            "api_key": api_key,
            "organization_id": organization_id,
            "rotation_period": rotation_period
        })
        return resp is not None and resp.status_code in (200, 204)

    def create_role(self, role_name: str, ttl: str = "1h", max_ttl: str = "24h") -> bool:
        """Create a Vault role for dynamic key generation."""
        resp = self._request("post", f"/v1/{self._engine_path}/roles/{role_name}", json={
            "ttl": ttl,
            "max_ttl": max_ttl
        })
        return resp is not None and resp.status_code in (200, 204)

    def get_dynamic_key(self, role_name: str) -> Dict[str, Any]:
        """
        Request a dynamic API key from Vault.
        Returns a short-lived API key that auto-expires.
        """
        resp = self._request("get", f"/v1/{self._engine_path}/creds/{role_name}")
        if resp is None or resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception:
            return None
        return {
            "api_key": data.get("data", {}).get("key"),
            "lease_id": data.get("lease_id"),
            "lease_duration": data.get("lease_duration"),
            "renewable": data.get("renewable"),
            "expire_time": data.get("data", {}).get("expire_time"),
            "provider": self._engine_path
        }

    def get_secret(self, key: str) -> str:
        """Get a static secret from Vault."""
        resp = self._request("get", f"/v1/secret/data/{key}")
        if resp is None or resp.status_code != 200:
            return None
        try:
            return resp.json().get("data", {}).get("data", {}).get("value")
        except Exception:
            return None

    def store_secret(self, key: str, value: str, ttl: int = None) -> bool:
        """Store a secret in Vault."""
        payload = {"data": {"value": value}}
        if ttl:
            payload["ttl"] = ttl
        resp = self._request("post", f"/v1/secret/metadata/{key}", json=payload)
        return resp is not None and resp.status_code in (200, 204)

    def delete_secret(self, key: str) -> bool:
        """Delete a secret from Vault."""
        resp = self._request("delete", f"/v1/secret/metadata/{key}")
        return resp is not None and resp.status_code == 204

    def list_secrets(self, prefix: str = "") -> List[str]:
        """List secrets in Vault."""
        resp = self._request("get", f"/v1/secret/metadata/?prefix={prefix}")
        if resp is None or resp.status_code != 200:
            return []
        try:
            return resp.json().get("data", {}).get("keys", [])
        except Exception:
            return []


class AWSSecretsManager(SecretProvider):
    """AWS Secrets Manager client."""

    def __init__(self, region: str = None, profile: str = None):
        import boto3
        self.region = region or os.environ.get('AWS_REGION', 'us-east-1')
        self.client = boto3.client(
            'secretsmanager',
            region_name=self.region,
            profile_name=profile
        )

    def get_secret(self, key: str) -> str:
        try:
            resp = self.client.get_secret_value(SecretId=key)
            return resp.get('SecretString')
        except Exception:
            return None

    def store_secret(self, key: str, value: str, ttl: int = None) -> bool:
        return False  # AWS SM is primarily for retrieval

    def delete_secret(self, key: str) -> bool:
        try:
            self.client.delete_secret(SecretId=key, RecoverVariable=False)
            return True
        except Exception:
            return False

    def list_secrets(self, prefix: str = "") -> List[str]:
        return []


class AzureKeyVault(SecretProvider):
    """Azure Key Vault client."""

    def __init__(self, vault_url: str = None):
        if not vault_url:
            raise ValueError("vault_url is required for AzureKeyVault")
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
        self.vault_url = vault_url or os.environ.get('AZURE_KEYVAULT_URL')
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=self.vault_url, credential=credential)

    def get_secret(self, key: str) -> str:
        try:
            secret = self.client.get_secret(key)
            return secret.value
        except Exception:
            return None

    def store_secret(self, key: str, value: str, ttl: int = None) -> bool:
        return False

    def delete_secret(self, key: str) -> bool:
        try:
            self.client.begin_delete_secret(key)
            return True
        except Exception:
            return False

    def list_secrets(self, prefix: str = "") -> List[str]:
        return []


def get_secret_provider(provider_type: str = "env", config: Dict[str, Any] = None) -> SecretProvider:
    """Factory function to get the appropriate secret provider.

    ``config`` is the provider specific section (e.g. ``vault: {...}``).
    Falls back to :class:`EnvSecretProvider` whenever the requested provider
    cannot be built, so callers always receive a usable provider.
    """
    cfg = config if isinstance(config, dict) else {}
    try:
        if provider_type == 'vault':
            url = cfg.get('url') or os.environ.get(cfg.get('url_env') or 'VAULT_ADDR')
            if not url:
                raise ValueError("vault url is not configured")
            return VaultSecretProvider(
                vault_url=url,
                token=cfg.get('token') or os.environ.get(cfg.get('token_env') or 'VAULT_TOKEN'),
                namespace=cfg.get('namespace'),
                engine=cfg.get('engine', 'openai'),
                timeout=float(cfg.get('timeout', 5.0)),
            )
        if provider_type == 'aws':
            return AWSSecretsManager(region=cfg.get('region'), profile=cfg.get('profile'))
        if provider_type == 'azure':
            url = cfg.get('url') or os.environ.get(cfg.get('vault_url_env') or cfg.get('url_env') or 'AZURE_KEYVAULT_URL')
            if not url:
                raise ValueError("azure key vault url is not configured")
            return AzureKeyVault(vault_url=url)
        if provider_type != 'env':
            warnings.warn(f"unknown secret provider {provider_type!r}, falling back to env")
        return EnvSecretProvider(env_file=cfg.get('env_file'))
    except Exception as exc:
        warnings.warn(f"cannot create {provider_type!r} secret provider ({exc}), falling back to env")
        return EnvSecretProvider()


def load_secrets_config(config_path: str = None) -> Dict[str, Any]:
    """Load ``config/secrets.yaml`` (or an explicit path); returns {} if missing."""
    if config_path is None:
        try:
            from providers.config_loader import get_config_loader
            return get_config_loader().load_secrets_config() or {}
        except Exception:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'secrets.yaml'
            )
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except OSError as exc:
        warnings.warn(f"cannot read secrets config {config_path}: {exc}")
        return {}


_SECRETS_KEYS = (
    'primary_provider', 'fallback_provider', 'fallback_providers',
    'vault', 'aws', 'azure', 'env',
)


def _extract_secrets_section(config: Any) -> Dict[str, Any]:
    """Accept either the yaml root (``{'secrets': {...}}``) or the section itself."""
    if not isinstance(config, dict):
        return {}
    nested = config.get('secrets')
    if isinstance(nested, dict):
        return nested
    if any(key in config for key in _SECRETS_KEYS):
        return config
    return {}


class SecretManager:
    """Centralized secret management with fallback chain."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config: Dict[str, Any] = load_secrets_config() if config is None else config
        self.secrets_config: Dict[str, Any] = _extract_secrets_section(self.config)
        self.providers: List[SecretProvider] = []
        self._setup_providers()

    def _setup_providers(self):
        primary = self.secrets_config.get('primary_provider', 'env')
        fallbacks = self.secrets_config.get('fallback_providers')
        if not isinstance(fallbacks, list):
            fallback = self.secrets_config.get('fallback_provider')
            fallbacks = [fallback] if fallback else []

        chain: List[str] = []
        for name in [primary] + list(fallbacks):
            if isinstance(name, str) and name and name not in chain:
                chain.append(name)
        if 'env' not in chain:
            chain.append('env')

        for name in chain:
            provider = self._build_provider(name)
            if provider is None:
                continue
            if any(isinstance(provider, type(existing)) for existing in self.providers):
                continue
            self.providers.append(provider)

    def _build_provider(self, name: str) -> Optional[SecretProvider]:
        section = self.secrets_config.get(name, {})
        if not isinstance(section, dict):
            section = {}
        return get_secret_provider(name, section)

    def get(self, key: str) -> str:
        for provider in self.providers:
            try:
                value = provider.get_secret(key)
            except Exception as exc:
                warnings.warn(f"{type(provider).__name__}.get_secret({key!r}) failed: {exc}")
                continue
            if value:
                return value
        return None

    def store(self, key: str, value: str) -> bool:
        for provider in self.providers:
            if not hasattr(provider, 'store_secret'):
                continue
            try:
                if provider.store_secret(key, value):
                    return True
            except Exception as exc:
                warnings.warn(f"{type(provider).__name__}.store_secret({key!r}) failed: {exc}")
        return False


def get_api_key(model: str = "openai", secret_manager: SecretManager = None) -> str:
    """Get API key for a model, prioritizing Vault dynamic keys."""
    manager = secret_manager or SecretManager()

    vault_cfg = manager.secrets_config.get('vault', {})
    if not isinstance(vault_cfg, dict):
        vault_cfg = {}
    openai_cfg = vault_cfg.get('openai_config', {})
    if not isinstance(openai_cfg, dict):
        openai_cfg = {}

    role = vault_cfg.get('role') or 'agent-rl-app'
    role_cfg = next(
        (r for r in vault_cfg.get('roles', []) if isinstance(r, dict) and r.get('name') == role),
        {}
    )
    admin_key = os.environ.get(openai_cfg.get('admin_key_env', 'OPENAI_ADMIN_KEY'), '')
    org_id = os.environ.get(openai_cfg.get('organization_id_env', 'OPENAI_ORG_ID'), '')

    for provider in manager.providers:
        if not isinstance(provider, VaultSecretProvider) or not provider.available:
            continue
        if not admin_key:
            warnings.warn("vault dynamic keys skipped: OPENAI_ADMIN_KEY is not set")
            break
        try:
            provider.enable_engine()
            provider.configure_engine(
                api_key=admin_key,
                organization_id=org_id,
                rotation_period=openai_cfg.get('rotation_period', '720h'),
            )
            provider.create_role(
                role,
                ttl=role_cfg.get('ttl', '1h'),
                max_ttl=role_cfg.get('max_ttl', '24h'),
            )
            result = provider.get_dynamic_key(role)
            if result and result.get('api_key'):
                return result['api_key']
        except Exception as exc:
            warnings.warn(f"vault dynamic key retrieval failed: {exc}")

    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_key = key_map.get(model, "OPENAI_API_KEY")
    key = os.environ.get(env_key)
    if key:
        return key
    try:
        from providers.config_loader import get_config_loader
        return get_config_loader().get_secret(env_key)
    except Exception:
        return None
