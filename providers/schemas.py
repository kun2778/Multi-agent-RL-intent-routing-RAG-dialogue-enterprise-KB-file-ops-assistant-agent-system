from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class SecretConfig:
    provider: str
    endpoint: str
    auth_method: str
    ttl: int = 3600
    max_ttl: int = 86400
    policies: List[str] = field(default_factory=list)


@dataclass
class ProviderConfig:
    name: str
    provider_type: str
    base_url: str
    auth: Dict[str, Any] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)
    routing: Dict[str, Any] = field(default_factory=dict)
    failover: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRoute:
    alias: str
    provider: str
    upstream_model: str
    weight: int = 1
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    auth_enabled: bool = True
    key_pool_size: int = 5
    max_retries: int = 3
    failover_timeout: int = 30
    usage_tracking: bool = True
    usage_db_path: str = "usage.db"
    rate_limit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLConfig:
    max_rounds: int = 5
    retrieval_budget: float = 300.0
    training_episodes: int = 10
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    gamma: float = 0.95
    learning_rate: float = 1e-3
    replay_buffer_size: int = 10000
    batch_size: int = 32


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    embedding_dim: int = 128
    similarity_threshold: float = 0.7
