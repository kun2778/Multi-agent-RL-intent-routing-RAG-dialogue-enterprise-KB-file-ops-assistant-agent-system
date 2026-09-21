import os
from typing import Dict, Any, List
from providers.base import LLMRequest, LLMResponse

class GeminiAdapter:
    def __init__(self, config: Dict[str, Any] = None):
        self.name = "gemini"
        self.base_url = config.get("base_url") or os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1")
        self.api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY", "")

    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(model_name=request.model)
            response = model.generate_content(request.messages)
            content = response.text
            return LLMResponse(content=content, model=request.model, provider=self.name)
        except Exception as e:
            return LLMResponse(content="", model=request.model, provider=self.name, error=str(e))

    def list_models(self) -> List[str]:
        return ["gemini-1.5-pro", "gemini-1.5-flash"]
