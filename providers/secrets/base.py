import os
import yaml
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

    def __init__(self, vault_url: str, token: str = None, namespace: str = None):
        self.vault_url = vault_url.rstrip('/')
        self.namespace = namespace
        self.token = token or os.environ.get('VAULT_TOKEN')
        self._client = None
        self._engine_path = 'openai'

    def _get_client(self):
        """Get or create a Vault HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                base_url=self.vault_url,
                headers={
                    "X-Vault-Token": self.token,
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            if self.namespace:
                self._client.headers["X-Vault-Namespace"] = self.namespace
        return self._client

    def enable_engine(self) -> bool:
        """Enable the OpenAI secrets engine in Vault."""
        client = self._get_client()
        try:
            resp = client.post(f"/v1/sys/plugins/catalog/{self._engine_path}", json={
                "type": "secret"
            })
            return resp.status_code == 204
        except Exception:
            return False

    def configure_engine(self, api_key: str, organization_id: str, rotation_period: str = "720h") -> bool:
        """Configure the OpenAI secrets engine."""
        client = self._get_client()
        try:
            resp = client.post(f"/v1/{self._engine_path}/config", json={
                "api_key": api_key,
                "organization_id": organization_id,
                "rotation_period": rotation_period
            })
            return resp.status_code == 204
        except Exception:
            return False

    def create_role(self, role_name: str, ttl: str = "1h", max_ttl: str = "24h") -> bool:
        """Create a Vault role for dynamic key generation."""
        client = self._get_client()
        try:
            resp = client.post(f"/v1/{self._engine_path}/roles/{role_name}", json={
                "ttl": ttl,
                "max_ttl": max_ttl
            })
            return resp.status_code == 204
        except Exception:
            return False

    def get_dynamic_key(self, role_name: str) -> Dict[str, Any]:
        """
        Request a dynamic API key from Vault.
        Returns a short-lived API key that auto-expires.
        """
        client = self._get_client()
        try:
            resp = client.get(f"/v1/{self._engine_path}/creds/{role_name}")
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "api_key": data.get("data", {}).get("key"),
                    "lease_id": data.get("lease_id"),
                    "lease_duration": data.get("lease_duration"),
                    "renewable": data.get("renewable"),
                    "expire_time": data.get("data", {}).get("expire_time"),
                    "provider": self._engine_path
                }
            return None
        except Exception:
            return None

    def get_secret(self, key: str) -> str:
        """Get a static secret from Vault."""
        client = self._get_client()
        try:
            resp = client.get(f"/v1/secret/data/{key}")
            if resp.status_code == 200:
                return resp.json().get("data", {}).get("data", {}).get("value")
        except Exception:
            pass
        return None

    def store_secret(self, key: str, value: str, ttl: int = None) -> bool:
        """Store a secret in Vault."""
        client = self._get_client()
        try:
            payload = {"data": {"value": value}}
            if ttl:
                payload["ttl"] = ttl
            resp = client.post(f"/v1/secret/metadata/{key}", json=payload)
            return resp.status_code in [200, 204]
        except Exception:
            return False

    def delete_secret(self, key: str) -> bool:
        """Delete a secret from Vault."""
        client = self._get_client()
        try:
            resp = client.delete(f"/v1/secret/metadata/{key}")
            return resp.status_code == 204
        except Exception:
            return False

    def list_secrets(self, prefix: str = "") -> List[str]:
        """List secrets in Vault."""
        client = self._get_client()
        try:
            resp = client.get(f"/v1/secret/metadata/?prefix={prefix}")
            if resp.status_code == 200:
                return resp.json().get("data", {}).get("keys", [])
        except Exception:
            pass
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


def get_secret_provider(provider_type: str = "env") -> SecretProvider:
    """Factory function to get the appropriate secret provider."""
    providers = {
        "env": EnvSecretProvider,
        "vault": VaultSecretProvider,
        "aws": AWSSecretsManager,
        "azure": AzureKeyVault,
    }
    provider_class = providers.get(provider_type, EnvSecretProvider)
    return provider_class()


class SecretManager:
    """Centralized secret management with fallback chain."""

    def __init__(self, config: Dict[str, Any] = None):
        self.providers: List[SecretProvider] = []
        self.config = config or {}
        self._setup_providers()

    def _setup_providers(self):
        secrets_config = self.config.get('secrets', {})
        primary = secrets_config.get('primary_provider', 'env')

        if primary == 'vault':
            vault_config = secrets_config.get('vault', {})
            self.providers.append(VaultSecretProvider(
                vault_url=vault_config.get('url'),
                token=vault_config.get('token')
            ))
        elif primary == 'aws':
            aws_config = secrets_config.get('aws', {})
            self.providers.append(AWSSecretsManager(region=aws_config.get('region')))
        elif primary == 'azure':
            azure_config = secrets_config.get('azure', {})
            self.providers.append(AzureKeyVault(vault_url=azure_config.get('url')))

        self.providers.append(EnvSecretProvider())

    def get(self, key: str) -> str:
        for provider in self.providers:
            value = provider.get_secret(key)
            if value:
                return value
        return None

    def store(self, key: str, value: str) -> bool:
        for provider in self.providers:
            if hasattr(provider, 'store_secret'):
                if provider.store_secret(key, value):
                    return True
        return False


def get_api_key(model: str = "openai") -> str:
    """Get API key for a model, prioritizing Vault dynamic keys."""
    secret_manager = SecretManager()

    vault_provider = None
    for p in secret_manager.providers:
        if isinstance(p, VaultSecretProvider):
            vault_provider = p
            break

    if vault_provider:
        try:
            vault_provider.enable_engine()
            vault_provider.configure_engine(
                api_key=os.environ.get('OPENAI_ADMIN_KEY', ''),
                organization_id=os.environ.get('OPENAI_ORG_ID', '')
            )
            vault_provider.create_role('agent-rl-app', ttl="1h", max_ttl="24h")
            result = vault_provider.get_dynamic_key('agent-rl-app')
            if result and result.get('api_key'):
                return result['api_key']
        except Exception:
            pass

    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_key = key_map.get(model, "OPENAI_API_KEY")
    return os.environ.get(env_key)
