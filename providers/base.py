from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class LLMRequest:
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.base_url = config.get("base_url", "")
        self.api_key = config.get("api_key", "")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 120)

    @abstractmethod
    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        """Generate chat completions."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models."""
        pass

    def health_check(self) -> bool:
        """Check if provider is healthy."""
        try:
            models = self.list_models()
            return len(models) > 0
        except Exception:
            return False

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"
