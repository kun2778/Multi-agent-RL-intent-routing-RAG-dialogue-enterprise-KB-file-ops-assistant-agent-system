"""线上数据脱敏（redaction）。

失败案例 / Trace 会带着真实查询、路径、答案落盘并回流进 replay buffer，
其中可能包含邮箱、手机号、密钥、本机用户名等敏感信息。
统一在**入口**脱敏，保证写盘与进 buffer 的都是脱敏后的内容。

注意：脱敏只做保守替换，不改变语句结构，避免破坏意图分类特征。
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["redact", "redact_path", "MAX_QUERY_LEN", "PLACEHOLDER"]

PLACEHOLDER = "<REDACTED>"
MAX_QUERY_LEN = 512   # 回流 buffer 的查询长度上限（超长截断，省内存也防 OOD）

_RULES = [
    # 邮箱
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), PLACEHOLDER),
    # Bearer token / sk- 开头的 key
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.I), f"Bearer {PLACEHOLDER}"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), f"sk-{PLACEHOLDER}"),
    # api_key/token/secret/password 赋值
    (re.compile(r"\b((?:api[_-]?key|access[_-]?token|token|secret|password|passwd)"
                r"\s*[:=]\s*)(\S+)", re.I), rf"\g<1>{PLACEHOLDER}"),
    # 中国大陆手机号
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), PLACEHOLDER),
    # IPv4
    (re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"), PLACEHOLDER),
    # 长十六进制 / base64 片段（可能是 hash、token）
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), PLACEHOLDER),
    (re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"), PLACEHOLDER),
]

# 路径里的本机用户名
_WIN_HOME = re.compile(r"(?i)([A-Za-z]:\\Users\\)[^\\\s\"']+")
_POSIX_HOME = re.compile(r"(/(?:home|Users)/)[^/\s\"']+")


def redact_path(text: str) -> str:
    """把绝对路径中的本机用户名替换成占位符。"""
    text = _WIN_HOME.sub(r"\g<1><USER>", text)
    text = _POSIX_HOME.sub(r"\g<1><USER>", text)
    return text


def redact(text: Optional[str], max_len: int = MAX_QUERY_LEN) -> str:
    """脱敏 + 截断。非字符串输入返回空串。"""
    if not text or not isinstance(text, str):
        return ""
    out = redact_path(text)
    for pattern, repl in _RULES:
        out = pattern.sub(repl, out)
    if len(out) > max_len:
        out = out[:max_len]
    return out
