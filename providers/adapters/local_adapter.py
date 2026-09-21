import os
from typing import Dict, Any, List
from providers.base import LLMRequest, LLMResponse

class LocalAdapter:
    def __init__(self, config: Dict[str, Any] = None):
        self.name = "local"
        self.base_url = config.get("base_url") or os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
        self.api_key = config.get("api_key", "local")

    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            resp = client.chat.completions.create(
                model=request.model, messages=request.messages,
                temperature=request.temperature, max_tokens=request.max_tokens,
            )
            content = resp.choices[0].message.content
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
            return LLMResponse(content=content, model=request.model, provider=self.name, usage=usage)
        except Exception as e:
            return LLMResponse(content="", model=request.model, provider=self.name, error=str(e))

    def list_models(self) -> List[str]:
        return ["llama2", "mistral", "codellama", "phi"]
