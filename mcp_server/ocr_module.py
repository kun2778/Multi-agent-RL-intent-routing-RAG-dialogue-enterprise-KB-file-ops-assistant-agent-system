"""OCR模块 - 支持多种OCR提供商的文本识别"""

import os
import base64
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    confidence: float
    language: str = "unknown"
    raw_data: Dict = None
    processing_time_ms: int = 0


class OCRProvider(ABC):
    """OCR提供商基类"""
    
    @abstractmethod
    def recognize(self, image_path: str = None, image_bytes: bytes = None) -> OCRResult:
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        pass


class BaiduOCR(OCRProvider):
    """百度OCR API"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.environ.get("BAIDU_OCR_API_KEY")
        self.secret_key = secret_key or os.environ.get("BAIDU_OCR_SECRET_KEY")
        self.access_token = None
        self._token_expires = 0
    
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)
    
    def _get_access_token(self) -> str:
        """获取百度API访问令牌"""
        import requests
        
        if self.access_token and time.time() < self._token_expires:
            return self.access_token
        
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        resp = requests.post(url, params=params, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        self.access_token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 2592000) - 3600
        
        return self.access_token
    
    def recognize(self, image_path: str = None, image_bytes: bytes = None) -> OCRResult:
        import requests
        import base64
        
        start_time = time.time()
        
        # 读取图像数据
        if image_bytes:
            image_data = image_bytes
        elif image_path:
            with open(image_path, "rb") as f:
                image_data = f.read()
        else:
            raise ValueError("Either image_path or image_bytes must be provided")
        
        # Base64编码
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # 调用百度OCR API
        token = self._get_access_token()
        url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"image": image_base64}
        
        resp = requests.post(url, params={"access_token": token}, 
                            headers=headers, data=data, timeout=30)
        resp.raise_for_status()
        
        result = resp.json()
        
        # 解析结果
        texts = []
        avg_confidence = 0.0
        
        if "words_result" in result:
            for item in result["words_result"]:
                texts.append(item["words"])
            avg_confidence = 0.9  # 百度API不直接返回置信度
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return OCRResult(
            text="\n".join(texts),
            confidence=avg_confidence,
            language="zh",
            raw_data=result,
            processing_time_ms=processing_time
        )


class TencentOCR(OCRProvider):
    """腾讯云OCR API"""
    
    def __init__(self, secret_id: str = None, secret_key: str = None):
        self.secret_id = secret_id or os.environ.get("TENCENT_OCR_SECRET_ID")
        self.secret_key = secret_key or os.environ.get("TENCENT_OCR_SECRET_KEY")
    
    def is_configured(self) -> bool:
        return bool(self.secret_id and self.secret_key)
    
    def recognize(self, image_path: str = None, image_bytes: bytes = None) -> OCRResult:
        # 腾讯OCR实现需要腾讯云SDK
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.ocr.v20161111 import ocr_client, models
            
            import time
            start_time = time.time()
            
            # 读取图像
            if image_bytes:
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            elif image_path:
                with open(image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
            else:
                raise ValueError("Either image_path or image_bytes must be provided")
            
            # 创建客户端
            cred = credential.Credential(self.secret_id, self.secret_key)
            httpProfile = HttpProfile()
            httpProfile.endpoint = "ocr.tencentcloudapi.com"
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            client = ocr_client.OcrClient(cred, "ap-beijing", clientProfile)
            
            # 调用通用OCR
            req = models.GeneralBasicOCRRequest()
            req.ImageBase64 = image_base64
            
            resp = client.GeneralBasicOCR(req)
            
            # 解析结果
            texts = []
            confidences = []
            
            for item in resp.TextDetections:
                texts.append(item.DetectedText)
                confidences.append(item.Confidence)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            processing_time = int((time.time() - start_time) * 1000)
            
            return OCRResult(
                text="\n".join(texts),
                confidence=avg_confidence,
                language="zh",
                processing_time_ms=processing_time
            )
            
        except ImportError:
            raise ImportError("Tencent Cloud SDK not installed. Run: pip install tencentcloud-sdk-python")


class TesseractOCR(OCRProvider):
    """Tesseract本地OCR（免费，无需API Key）"""
    
    def __init__(self, language: str = "chi_sim+eng"):
        self.language = language
        self._available = None
    
    def is_configured(self) -> bool:
        if self._available is None:
            try:
                import pytesseract
                # 测试tesseract是否可用
                pytesseract.get_tesseract_version()
                self._available = True
            except (ImportError, Exception):
                self._available = False
        return self._available
    
    def recognize(self, image_path: str = None, image_bytes: bytes = None) -> OCRResult:
        import pytesseract
        from PIL import Image
        import io
        
        start_time = time.time()
        
        # 加载图像
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes))
        elif image_path:
            image = Image.open(image_path)
        else:
            raise ValueError("Either image_path or image_bytes must be provided")
        
        # 执行OCR
        text = pytesseract.image_to_string(image, lang=self.language)
        
        # 获取详细信息
        data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
        
        # 计算平均置信度
        confidences = [int(c) for c in data["conf"] if int(c) > 0]
        avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return OCRResult(
            text=text.strip(),
            confidence=avg_confidence,
            language=self.language,
            processing_time_ms=processing_time
        )


class OCRManager:
    """OCR管理器 - 统一管理多个OCR提供商"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_config()
        self.providers: Dict[str, OCRProvider] = {}
        self._init_providers()
    
    def _load_config(self) -> Dict:
        """加载配置"""
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "app.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("ocr", {})
        return {}
    
    def _init_providers(self):
        """初始化OCR提供商"""
        provider = self.config.get("provider", "tesseract")
        
        # 百度OCR
        if provider == "baidu" or "baidu" in str(self.config):
            self.providers["baidu"] = BaiduOCR()
        
        # 腾讯OCR
        if provider == "tencent" or "tencent" in str(self.config):
            self.providers["tencent"] = TencentOCR()
        
        # Tesseract（始终可用作为备选）
        self.providers["tesseract"] = TesseractOCR(
            language=self.config.get("tesseract", {}).get("language", "chi_sim+eng")
        )
    
    def get_provider(self, name: str = None) -> Optional[OCRProvider]:
        """获取OCR提供商"""
        if name:
            provider = self.providers.get(name)
            if provider and provider.is_configured():
                return provider
            return None
        
        # 按优先级查找
        preferred_order = [self.config.get("provider", ""), "baidu", "tencent", "tesseract"]
        for name in preferred_order:
            provider = self.providers.get(name)
            if provider and provider.is_configured():
                return provider
        
        return None
    
    def recognize(self, image_path: str = None, image_bytes: bytes = None,
                  provider: str = None) -> Optional[OCRResult]:
        """执行OCR识别"""
        ocr_provider = self.get_provider(provider)
        if not ocr_provider:
            raise RuntimeError("No configured OCR provider available")
        
        return ocr_provider.recognize(image_path=image_path, image_bytes=image_bytes)
    
    def get_available_providers(self) -> List[str]:
        """获取可用的OCR提供商列表"""
        return [name for name, provider in self.providers.items() 
                if provider.is_configured()]


# 全局OCR管理器实例
_ocr_manager = None


def get_ocr_manager() -> OCRManager:
    """获取全局OCR管理器"""
    global _ocr_manager
    if _ocr_manager is None:
        _ocr_manager = OCRManager()
    return _ocr_manager


def ocr_recognize(image_path: str = None, image_bytes: bytes = None,
                  provider: str = None) -> Optional[str]:
    """便捷函数：执行OCR识别并返回文本"""
    manager = get_ocr_manager()
    result = manager.recognize(image_path=image_path, image_bytes=image_bytes, 
                               provider=provider)
    return result.text if result else None
