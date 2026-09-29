"""PDF 解析模块：提取 PDF 文本，供扫描注册、索引分块与内容检索使用。

后端优先级（任一可用即返回，全部失败返回 None）：
  1. pypdf / PyPDF2
  2. pdfplumber
  3. PyMuPDF (fitz)
  4. 内置简易解析（zlib 解压内容流 + 文本操作符提取，尽力而为）

对外接口：
  - is_pdf(path) -> bool
  - parse_pdf(path) -> PDFParseResult（text / pages / backend / error）
  - extract_pdf_text(path) -> Optional[str]
"""

import os
import re
import zlib
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PDFParseResult:
    """PDF 解析结果。"""
    text: str = ""
    pages: int = 0
    backend: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def is_pdf(path: str) -> bool:
    """判断路径是否为 PDF 文件（按扩展名）。"""
    return os.path.splitext(path or "")[1].lower() == ".pdf"


# ──────────────────────────────────────────────────────────────────────
# 各解析后端
# ──────────────────────────────────────────────────────────────────────

def _parse_with_pypdf(path: str, max_chars: Optional[int] = None) -> Optional[PDFParseResult]:
    Reader = None
    try:
        from pypdf import PdfReader as Reader  # noqa: N813
    except ImportError:
        try:
            from PyPDF2 import PdfReader as Reader  # noqa: N813
        except ImportError:
            return None
    reader = Reader(path)
    pages: List[str] = []
    total = 0
    truncated = False
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
        total += len(t)
        # 流式逐页：够用了就停，不为截断目的去解压整份 PDF
        if max_chars and max_chars > 0 and total >= max_chars:
            truncated = True
            break
    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        return PDFParseResult(text="", pages=len(pages),
                              backend="pypdf", error="未提取到文本（可能为扫描件）")
    result = PDFParseResult(text=text, pages=len(pages), backend="pypdf")
    if truncated:
        result.metadata["truncated"] = True
    return result


def _parse_with_pdfplumber(path: str, max_chars: Optional[int] = None) -> Optional[PDFParseResult]:
    try:
        import pdfplumber
    except ImportError:
        return None
    pages: List[str] = []
    total = 0
    truncated = False
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
            total += len(t)
            if max_chars and max_chars > 0 and total >= max_chars:
                truncated = True
                break
    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        return PDFParseResult(text="", pages=len(pages),
                              backend="pdfplumber", error="未提取到文本（可能为扫描件）")
    result = PDFParseResult(text=text, pages=len(pages), backend="pdfplumber")
    if truncated:
        result.metadata["truncated"] = True
    return result


def _parse_with_fitz(path: str, max_chars: Optional[int] = None) -> Optional[PDFParseResult]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    doc = fitz.open(path)
    try:
        pages: List[str] = []
        total = 0
        truncated = False
        for page in doc:            # 迭代式取页，不一次性铺开
            t = page.get_text() or ""
            pages.append(t)
            total += len(t)
            if max_chars and max_chars > 0 and total >= max_chars:
                truncated = True
                break
    finally:
        doc.close()
    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        return PDFParseResult(text="", pages=len(pages),
                              backend="fitz", error="未提取到文本（可能为扫描件）")
    result = PDFParseResult(text=text, pages=len(pages), backend="fitz")
    if truncated:
        result.metadata["truncated"] = True
    return result


_PDF_STRING_RE = re.compile(
    rb"\(((?:\\.|[^\\()])*)\)", re.S)


def _unescape_pdf_string(raw: bytes) -> str:
    """还原 PDF 字符串转义（\\n \\t \\( \\) \\\\ 及八进制）。"""
    out = bytearray()
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c != 0x5C:  # '\'
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        nxt = raw[i]
        mapping = {0x6E: 0x0A, 0x72: 0x0D, 0x74: 0x09,
                   0x62: 0x08, 0x66: 0x0C,
                   0x28: 0x28, 0x29: 0x29, 0x5C: 0x5C}
        if nxt in mapping:
            out.append(mapping[nxt])
            i += 1
        elif 0x30 <= nxt <= 0x37:  # 八进制转义
            j = i
            while j < n and j < i + 3 and 0x30 <= raw[j] <= 0x37:
                j += 1
            try:
                out.append(int(raw[i:j], 8) & 0xFF)
            except ValueError:
                pass
            i = j
        else:
            out.append(nxt)
            i += 1
    return out.decode("utf-8", errors="ignore")


def _parse_builtin(path: str, max_chars: Optional[int] = None) -> Optional[PDFParseResult]:
    """内置简易解析：解压内容流并提取 Tj/TJ 文本操作符中的字符串。"""
    try:
        from atomic_io import read_bytes
        data = read_bytes(path)
    except OSError as e:
        return PDFParseResult(error=f"读取失败: {e}")
    if not data.startswith(b"%PDF"):
        return PDFParseResult(error="非有效 PDF 文件")

    fragments: List[str] = []
    total = 0
    truncated = False
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = m.group(1)
        # FlateDecode 压缩流解压；未压缩流直接使用
        try:
            raw = zlib.decompress(raw)
        except (zlib.error, OSError):
            pass
        # 仅处理含文本操作符的流
        if b"Tj" not in raw and b"TJ" not in raw and b"BT" not in raw:
            continue
        for sm in _PDF_STRING_RE.finditer(raw):
            s = _unescape_pdf_string(sm.group(1)).strip()
            if s:
                fragments.append(s)
                total += len(s)
        fragments.append("\n")
        # 流式截断：够了就不再继续扫后续内容流
        if max_chars and max_chars > 0 and total >= max_chars:
            truncated = True
            break

    text = "".join(fragments)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return PDFParseResult(pages=0, backend="builtin",
                              error="内置解析未提取到文本（可能为压缩/扫描件）")
    pages = data.count(b"/Type /Page") or data.count(b"/Type/Page")
    result = PDFParseResult(text=text, pages=pages, backend="builtin")
    if truncated:
        result.metadata["truncated"] = True
    return result


# ──────────────────────────────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────────────────────────────

def parse_pdf(path: str, max_chars: Optional[int] = None) -> PDFParseResult:
    """解析 PDF，按后端优先级尝试，返回 PDFParseResult。

    各后端均为**流式逐页**处理：传入 ``max_chars`` 时，累计字符数达到阈值
    即停止取下一页（``metadata['truncated'] = True``），不会为了截断而把
    整份 PDF 解压进内存。

    Args:
        path: PDF 文件路径
        max_chars: 截断上限（None 不截断，逐页处理到末页）
    """
    if not path or not os.path.exists(path):
        return PDFParseResult(error=f"文件不存在: {path}")
    if not is_pdf(path):
        return PDFParseResult(error="非 .pdf 扩展名")

    last: Optional[PDFParseResult] = None
    for backend in (_parse_with_pypdf, _parse_with_pdfplumber,
                    _parse_with_fitz, _parse_builtin):
        try:
            result = backend(path, max_chars)
        except Exception as e:
            last = PDFParseResult(error=str(e))
            continue
        if result is None:
            continue
        if result.ok:
            if max_chars is not None and max_chars > 0:
                result.text = result.text[:max_chars]
            return result
        last = result
    return last or PDFParseResult(error="无可用 PDF 解析后端")


def extract_pdf_text(path: str, max_chars: Optional[int] = None) -> Optional[str]:
    """提取 PDF 文本；失败返回 None（调用方自行回退）。"""
    result = parse_pdf(path, max_chars=max_chars)
    if result.ok:
        return result.text
    return None
