import os
import json
from typing import Dict, Any, List, Optional
from providers.base import LLMRequest, LLMResponse


class OpenAIAdapter:
    """Adapter for OpenAI API."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = "openai"
        self.base_url = config.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 120)

    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            kwargs = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "stream": request.stream,
            }
            if request.extra_params:
                kwargs.update(request.extra_params)

            resp = client.chat.completions.create(**kwargs)

            if request.stream:
                content = ""
                for chunk in resp:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                }
                return LLMResponse(content=content, model=request.model, provider=self.name, usage=usage)
            else:
                content = resp.choices[0].message.content
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                }
                return LLMResponse(content=content, model=request.model, provider=self.name, usage=usage,
                                   raw_response=resp.model_dump())
        except Exception as e:
            return LLMResponse(content="", model=request.model, provider=self.name, error=str(e))

    def list_models(self) -> List[str]:
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        return models


class AnthropicAdapter:
    """Adapter for Anthropic Claude API."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = "anthropic"
        self.base_url = config.get("base_url") or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        self.api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")

    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            resp = client.messages.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            content = "".join([block.text for block in resp.content if block.type == "text"])
            usage = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            }
            return LLMResponse(content=content, model=request.model, provider=self.name, usage=usage)
        except Exception as e:
            return LLMResponse(content="", model=request.model, provider=self.name, error=str(e))

    def list_models(self) -> List[str]:
        return ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]


class DeepSeekAdapter:
    """Adapter for DeepSeek API."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = "deepseek"
        self.base_url = config.get("base_url") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")

    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            resp = client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            content = resp.choices[0].message.content
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            return LLMResponse(content=content, model=request.model, provider=self.name, usage=usage)
        except Exception as e:
            return LLMResponse(content="", model=request.model, provider=self.name, error=str(e))

    def list_models(self) -> List[str]:
        return ["deepseek-chat", "deepseek-coder"]


class GeminiAdapter:
    """Adapter for Google Gemini API."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = "gemini"
        self.base_url = config.get("base_url") or os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1")
        self.api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY", "")

    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(model_name=request.model)
            response = model.generate_content(
                request.messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                )
            )
            content = response.text
            return LLMResponse(content=content, model=request.model, provider=self.name)
        except Exception as e:
            return LLMResponse(content="", model=request.model, provider=self.name, error=str(e))

    def list_models(self) -> List[str]:
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]


class LocalAdapter:
    """Adapter for local LLM models (Ollama, llama.cpp, etc.)."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = "local"
        self.base_url = config.get("base_url") or os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
        self.api_key = config.get("api_key", "local")

    def chat_completions(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            resp = client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            content = resp.choices[0].message.content
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            return LLMResponse(content=content, model=request.model, provider=self.name, usage=usage)
        except Exception as e:
            return LLMResponse(content="", model=request.model, provider=self.name, error=str(e))

    def list_models(self) -> List[str]:
        return ["llama2", "mistral", "codellama", "phi"]
