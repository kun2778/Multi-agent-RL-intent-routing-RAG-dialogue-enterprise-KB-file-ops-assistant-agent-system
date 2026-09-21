"""OCR 处理器 - 实现图片文字识别和最近截屏查找功能。

功能：
1. 查找最近的截屏文件
2. 调用OCR识别图片文字
3. 结合文件内容输出结果
"""

import os
import glob
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from agent_state import AgentState, QuerySubIntent, SlotType, RetrievalResult


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    confidence: float
    source_file: str
    processing_time_ms: int = 0


class ScreenshotFinder:
    """查找最近的截屏文件。"""

    SCREENSHOT_DIRS = [
        os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots"),
        os.path.join(os.path.expanduser("~"), "Pictures"),
        os.path.join(os.path.expanduser("~"), "Desktop"),
        os.path.join(os.path.expanduser("~"), "Documents"),
        os.path.join(os.path.expanduser("~"), "Downloads"),
    ]

    SCREENSHOT_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]

    SCREENSHOT_KEYWORDS = [
        "screenshot", "截屏", "截图", "screen", "capture",
        "snip", "剪贴板", "clipboard","图片"
    ]

    def find_recent_screenshots(self, hours: int = 24, max_count: int = 5) -> List[Dict[str, Any]]:
        """查找最近N小时内的截屏文件。"""
        screenshots = []
        cutoff_time = time.time() - (hours * 3600)

        for search_dir in self.SCREENSHOT_DIRS:
            if not os.path.exists(search_dir):
                continue

            for ext in self.SCREENSHOT_EXTENSIONS:
                pattern = os.path.join(search_dir, "**", ext)
                for filepath in glob.glob(pattern, recursive=True):
                    try:
                        stat = os.stat(filepath)
                        if stat.st_mtime >= cutoff_time:
                            screenshots.append({
                                "path": filepath,
                                "name": os.path.basename(filepath),
                                "modified_time": stat.st_mtime,
                                "size": stat.st_size,
                                "is_screenshot": self._is_screenshot_file(filepath),
                            })
                    except (OSError, PermissionError):
                        continue

        screenshots.sort(key=lambda x: x["modified_time"], reverse=True)
        return screenshots[:max_count]

    def find_latest_screenshot(self, find_oldest: bool = False) -> Optional[str]:
        """查找最新或最早的截屏文件。"""
        screenshots_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots")
        if os.path.exists(screenshots_dir):
            screenshots = []
            for ext in self.SCREENSHOT_EXTENSIONS:
                try:
                    files = os.listdir(screenshots_dir)
                    for filename in files:
                        if filename.lower().endswith(ext[1:]):
                            filepath = os.path.join(screenshots_dir, filename)
                            try:
                                stat = os.stat(filepath)
                                screenshots.append({
                                    "path": filepath,
                                    "modified_time": stat.st_mtime,
                                })
                            except (OSError, PermissionError):
                                continue
                except OSError:
                    continue
            if screenshots:
                screenshots.sort(key=lambda x: x["modified_time"], reverse=not find_oldest)
                return screenshots[0]["path"]
        
        screenshots = self.find_recent_screenshots(hours=8760, max_count=10)
        if screenshots:
            screenshots.sort(key=lambda x: x["modified_time"], reverse=not find_oldest)
            return screenshots[0]["path"]
        return None

    def _is_screenshot_file(self, filepath: str) -> bool:
        """判断文件是否为截屏文件（基于文件名关键词）。"""
        filename = os.path.basename(filepath).lower()
        return any(kw in filename for kw in self.SCREENSHOT_KEYWORDS)

    def find_screenshot_by_time_range(self, time_range: str) -> List[Dict[str, Any]]:
        """根据时间范围查找截屏。"""
        hours_map = {
            "recent": 4,
            "today": 24,
            "yesterday": 48,
            "this_week": 168,
            "this_month": 720,
        }
        hours = hours_map.get(time_range, 24)
        return self.find_recent_screenshots(hours=hours)


class OCRHandler:
    """OCR处理器 - 整合截屏查找和文字识别。"""

    def __init__(self):
        self.screenshot_finder = ScreenshotFinder()
        self._ocr_manager = None

    def _get_ocr_manager(self):
        """延迟加载OCR管理器。"""
        if self._ocr_manager is None:
            try:
                from mcp_server.ocr_module import get_ocr_manager
                self._ocr_manager = get_ocr_manager()
            except ImportError:
                print("  [OCR] 警告: 无法加载OCR模块，将使用本地Tesseract")
                self._ocr_manager = self._create_local_ocr()
        return self._ocr_manager

    def _create_local_ocr(self):
        """创建本地Tesseract OCR作为备选。"""
        try:
            from mcp_server.ocr_module import TesseractOCR, OCRManager
            tesseract = TesseractOCR()
            manager = OCRManager.__new__(OCRManager)
            manager.providers = {"tesseract": tesseract}
            manager.config = {"provider": "tesseract"}
            return manager
        except Exception:
            return None

    def process_ocr_request(self, state: AgentState, query: str) -> str:
        """处理OCR识别请求。

        流程：
        1. 从槽位或查询中获取图片路径
        2. 如果没有指定路径，查找最近或最早的截屏
        3. 调用OCR识别文字
        4. 返回识别结果
        """
        slots = state.slots
        image_path = slots.get(SlotType.IMAGE_PATH.value)

        if not image_path:
            if self._is_recent_screenshot_query(query):
                find_oldest = "最早" in query or "earliest" in query.lower() or "first" in query.lower()
                image_path = self.screenshot_finder.find_latest_screenshot(find_oldest=find_oldest)
                if image_path:
                    time_desc = "最早" if find_oldest else "最近"
                    print(f"  [OCR] 自动定位到{time_desc}截屏: {image_path}")
                else:
                    return "未找到截屏文件。请指定图片路径或检查截图目录。"
            else:
                return "请指定要识别的图片文件路径。"

        if not os.path.exists(image_path):
            return f"图片文件不存在: {image_path}"

        result = self._perform_ocr(image_path)
        if result is None:
            return f"OCR识别失败。请检查图片文件是否有效: {image_path}"

        return self._format_ocr_result(result, query)

    def _is_recent_screenshot_query(self, query: str) -> bool:
        """判断查询是否在询问最近或最早的截屏。"""
        q = query.lower()
        time_keywords = ["最近", "刚才", "最新", "刚截", "recent", "latest", "最早", "earliest", "first"]
        screenshot_keywords = ["截屏", "截图", "屏幕", "screen", "screenshot"]
        return any(rk in q for rk in time_keywords) and any(sk in q for sk in screenshot_keywords)

    def _perform_ocr(self, image_path: str) -> Optional[OCRResult]:
        """执行OCR识别。"""
        manager = self._get_ocr_manager()
        if manager is None:
            return self._fallback_tesseract(image_path)

        try:
            start_time = time.time()
            result = manager.recognize(image_path=image_path)
            processing_time = int((time.time() - start_time) * 1000)

            if result and result.text:
                return OCRResult(
                    text=result.text,
                    confidence=result.confidence,
                    source_file=image_path,
                    processing_time_ms=processing_time,
                )
        except Exception as e:
            print(f"  [OCR] 识别失败: {e}")
            return self._fallback_tesseract(image_path)

        return None

    def _fallback_tesseract(self, image_path: str) -> Optional[OCRResult]:
        """使用Tesseract作为备选OCR。"""
        try:
            import pytesseract
            from PIL import Image
            import time

            start_time = time.time()
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")
            processing_time = int((time.time() - start_time) * 1000)

            if text.strip():
                return OCRResult(
                    text=text.strip(),
                    confidence=0.8,
                    source_file=image_path,
                    processing_time_ms=processing_time,
                )
        except Exception as e:
            print(f"  [OCR] Tesseract识别失败: {e}")

        return None

    def _format_ocr_result(self, result: OCRResult, query: str) -> str:
        """格式化OCR识别结果。"""
        lines = [
            f"  [OCR识别结果]",
            f"  来源文件: {result.source_file}",
            f"  识别置信度: {result.confidence:.1%}",
            f"  处理时间: {result.processing_time_ms}ms",
            f"  ",
            f"  识别文字:",
            f"  {'─' * 40}",
        ]

        for line in result.text.split('\n'):
            lines.append(f"  {line}")

        lines.append(f"  {'─' * 40}")
        lines.append(f"  共识别 {len(result.text)} 个字符")

        return "\n".join(lines)

    def find_and_ocr_recent_screenshot(self, hours: int = 24) -> str:
        """查找并识别最近的截屏。"""
        screenshots = self.screenshot_finder.find_recent_screenshots(hours=hours)

        if not screenshots:
            return f"最近 {hours} 小时内未找到截屏文件。"

        results = []
        for i, ss in enumerate(screenshots[:3], 1):
            result = self._perform_ocr(ss["path"])
            if result:
                results.append({
                    "file": ss["name"],
                    "path": ss["path"],
                    "text": result.text,
                    "confidence": result.confidence,
                })

        if not results:
            return "找到截屏文件，但OCR识别失败。"

        output_lines = [f"  找到 {len(screenshots)} 个最近截屏，识别结果如下：\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(f"  [{i}] {r['file']}")
            output_lines.append(f"      路径: {r['path']}")
            output_lines.append(f"      置信度: {r['confidence']:.1%}")
            output_lines.append(f"      文字内容:")
            for line in r['text'].split('\n')[:5]:
                output_lines.append(f"        {line}")
            if len(r['text'].split('\n')) > 5:
                output_lines.append(f"        ... (共 {len(r['text'])} 字符)")
            output_lines.append("")

        return "\n".join(output_lines)

    def get_screenshot_info(self) -> str:
        """获取最近截屏文件信息。"""
        screenshots = self.screenshot_finder.find_recent_screenshots(hours=48)

        if not screenshots:
            return "最近 48 小时内未找到截屏文件。"

        lines = [f"  最近截屏文件 ({len(screenshots)} 个):\n"]
        for i, ss in enumerate(screenshots, 1):
            mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ss["modified_time"]))
            size_kb = ss["size"] / 1024
            lines.append(f"  [{i}] {ss['name']}")
            lines.append(f"      路径: {ss['path']}")
            lines.append(f"      修改时间: {mod_time}")
            lines.append(f"      大小: {size_kb:.1f} KB")
            lines.append(f"      可能是截屏: {'是' if ss['is_screenshot'] else '否'}")
            lines.append("")

        return "\n".join(lines)


def create_ocr_handler() -> OCRHandler:
    """创建OCR处理器实例。"""
    return OCRHandler()
