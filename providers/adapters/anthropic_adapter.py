import os
from typing import Dict, Any, List
from providers.base import LLMRequest, LLMResponse


class AnthropicAdapter:
    def __init__(self, config: Dict[str, Any] = None):
        self.name = "anthropic"
        self.base_url = config.get("base_url") or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        self.api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")

    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            resp = client.messages.create(
                model=request.model, messages=request.messages,
                temperature=request.temperature, max_tokens=request.max_tokens,
            )
            content = "".join([block.text for block in resp.content if block.type == "text"])
            usage = {"prompt_tokens": resp.usage.input_tokens, "completion_tokens": resp.usage.output_tokens}
            return LLMResponse(content=content, model=request.model, provider=self.name, usage=usage)
        except Exception as e:
            return LLMResponse(content="", model=request.model, provider=self.name, error=str(e))

    def list_models(self) -> List[str]:
        return ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]