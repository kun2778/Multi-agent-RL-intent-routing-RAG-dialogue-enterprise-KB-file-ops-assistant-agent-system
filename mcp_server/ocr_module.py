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
    provider: str = ""


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
        
        # 读取图像数据并 Base64 编码（大图走只读 mmap，不整块读进内存）
        if image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        elif image_path:
            from atomic_io import open_read
            with open_read(image_path) as image_data:
                image_base64 = base64.b64encode(image_data).decode("utf-8")
        else:
            raise ValueError("Either image_path or image_bytes must be provided")
        
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
            
            # 读取图像（大图走只读 mmap，不整块读进内存）
            if image_bytes:
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            elif image_path:
                from atomic_io import open_read
                with open_read(image_path) as image_data:
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
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
        
        # 加载图像（BytesIO 分级：小图直接 BytesIO，大图落临时文件避免复制缓冲区）
        if image_bytes:
            from atomic_io import open_blob
            with open_blob(image_bytes, suffix=".img") as stream:
                image = Image.open(stream)
                image.load()          # 立即解码，之后流可以安全关闭
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


class RapidOCRProvider(OCRProvider):
    """RapidOCR本地OCR（PP-OCRv4，免费离线，进程内，速度与质量均衡的主力）"""

    _engine = None
    _available = None

    def is_configured(self) -> bool:
        if self._available is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # noqa: F401
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def recognize(self, image_path: str = None, image_bytes: bytes = None) -> OCRResult:
        import numpy as np
        import time as _time

        engine = self._get_engine()
        start = _time.time()

        # RapidOCR 接受文件路径或 numpy array
        target = image_path
        if image_bytes is not None:
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            target = np.array(img)

        result, _elapse = engine(target)
        processing_time = int((_time.time() - start) * 1000)

        texts = []
        confidences = []
        if result:
            for item in result:
                # item = [bbox, text, conf]
                if len(item) >= 3:
                    texts.append(item[1])
                    confidences.append(float(item[2]))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text="\n".join(texts),
            confidence=avg_conf,
            language="zh+en",
            processing_time_ms=processing_time,
        )


class PaddleOCRVLProvider(OCRProvider):
    """PaddleOCR-VL（0.9B VLM，准但慢，作为低置信度兜底）。

    通过 subprocess 调用独立 venv 中的常驻 worker（paddle_ocr_worker.py），
    模型只加载一次，复用进程。CPU 单图约数分钟，仅在兜底时触发。
    """

    def __init__(self, worker_py: str = None, venv_python: str = None,
                 startup_timeout: float = 300.0, request_timeout: float = 900.0):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.worker_script = worker_py or os.path.join(base_dir, "paddle_ocr_worker.py")
        self.venv_python = venv_python or os.path.join(base_dir, "venv-paddle", "Scripts", "python.exe")
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._proc = None

    def is_configured(self) -> bool:
        return os.path.exists(self.venv_python) and os.path.exists(self.worker_script)

    def _ensure_worker(self):
        import subprocess
        if self._proc is not None and self._proc.poll() is None:
            return
        # 启动常驻 worker（stdin/stdout 管道）
        self._proc = subprocess.Popen(
            [self.venv_python, "-X", "utf8", self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        # 等待 worker 就绪：发 ping 直到收到 pong
        import select
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            self._proc.stdin.write(json.dumps({"cmd": "ping"}) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                break
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("pong"):
                return
        raise RuntimeError("PaddleOCR-VL worker 启动超时")

    def recognize(self, image_path: str = None, image_bytes: bytes = None) -> OCRResult:
        if not image_path:
            raise ValueError("PaddleOCR-VLProvider 仅支持 image_path")
        self._ensure_worker()

        start = time.time()
        self._proc.stdin.write(json.dumps({"image_path": image_path}, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        processing_time = int((time.time() - start) * 1000)

        if not line:
            self._proc = None  # worker 挂了，下次重启
            raise RuntimeError("PaddleOCR-VL worker 无响应")

        resp = json.loads(line)
        if not resp.get("ok"):
            self._proc = None
            raise RuntimeError(resp.get("error", "PaddleOCR-VL 识别失败"))

        return OCRResult(
            text=resp.get("text", ""),
            confidence=float(resp.get("confidence", 0.0)),
            language="zh+en",
            processing_time_ms=processing_time,
        )

    def shutdown(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None


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
        provider = self.config.get("provider", "rapidocr")
        
        # 百度OCR
        if provider == "baidu" or "baidu" in str(self.config):
            self.providers["baidu"] = BaiduOCR()
        
        # 腾讯OCR
        if provider == "tencent" or "tencent" in str(self.config):
            self.providers["tencent"] = TencentOCR()
        
        # RapidOCR（主力，进程内，免费离线）
        self.providers["rapidocr"] = RapidOCRProvider()
        
        # PaddleOCR-VL（兜底，慢但准）
        if self.config.get("paddleocr_vl", {}).get("enabled", True):
            self.providers["paddleocr_vl"] = PaddleOCRVLProvider()
        
        # Tesseract（最后备选）
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
        
        # 主链优先级：配置指定 > rapidocr > 云 > tesseract
        preferred_order = [self.config.get("provider", ""), "rapidocr", "baidu", "tencent", "tesseract"]
        for pname in preferred_order:
            if not pname:
                continue
            provider = self.providers.get(pname)
            if provider and provider.is_configured():
                return provider
        
        return None
    
    def recognize(self, image_path: str = None, image_bytes: bytes = None,
                  provider: str = None) -> Optional[OCRResult]:
        """执行OCR识别"""
        ocr_provider = self.get_provider(provider)
        if not ocr_provider:
            raise RuntimeError("No configured OCR provider available")
        
        result = ocr_provider.recognize(image_path=image_path, image_bytes=image_bytes)
        if result is not None and not result.provider:
            # 标注实际使用的 provider 名
            for pname, pobj in self.providers.items():
                if pobj is ocr_provider:
                    result.provider = pname
                    break
        return result
    
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
