from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


# ---- Gateway Request/Response Models ----

class ChatRequest(BaseModel):
    model: str = "gpt-4o"
    messages: List[Dict[str, str]] = Field(..., min_items=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: datetime = Field(default_factory=datetime.utcnow)
    model: str
    choices: List[Dict] = Field(default_factory=list)
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, str]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: Dict[str, str]
    message: str


# ---- Usage Models ----

class UsageRecord(BaseModel):
    gateway_key: str
    user_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    upstream_provider: str
    response_time_ms: int
    created_at: datetime


class UsageSummary(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    request_count: int
    models_used: List[str]


# ---- Key Models ----

class GatewayKey(BaseModel):
    key: str
    user_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    active: bool
    request_count: int
    max_requests: int


class KeyCreateRequest(BaseModel):
    user_id: str
    expires_days: int = 90
    max_requests: int = -1
    rate_limit: int = 0


class KeyCreateResponse(BaseModel):
    key: str
    user_id: str
    expires_at: datetime
    message: str


# ---- Provider Models ----

class ProviderInfo(BaseModel):
    name: str
    provider_type: str
    healthy: bool
    models: List[str]
    latency_ms: int


class ProviderStatus(BaseModel):
    providers: List[ProviderInfo]
    total_models: int
    healthy_count: int
    unhealthy_count: int