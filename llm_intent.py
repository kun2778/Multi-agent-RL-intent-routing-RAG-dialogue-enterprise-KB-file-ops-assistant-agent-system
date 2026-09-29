"""LLM 意图分类器：用大模型做自然语言 -> 操作意图 的语义路由。

设计：
  - 零第三方依赖：通过 urllib 直接调用 OpenAI 兼容端点 /chat/completions
    （openai / deepseek / local(Ollama) 均兼容），anthropic、gemini 用其原生 HTTP 接口。
  - 配置来源：config/app.yaml 的 intent_llm 段（provider/model/timeout）；
    base_url 与密钥变量名来自 config/models.yaml 的 providers 段。
  - 端点链容灾：intent_llm.fallback 段可配置备用端点（如本地 Ollama 的
    qwen2.5:3b-instruct）。主端点密钥缺失 / 密钥无效 / 调用失败 / 结果解析失败时，
    自动依次切换到后备端点；全部失败返回 None，由 AgentRL 回退到关键词规则 /
    Q 网络，保证系统不中断。
"""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from providers.config_loader import get_config_loader

VALID_INTENTS = [
    "add_file", "delete_file", "modify_file",
    "refactor", "query", "deep_search", "vague",
    "file_analysis", "file_scan", "report_generate",
    "filename_query",
]

VALID_QUERY_SUBINTENTS = [
    "query_metadata", "query_content", "query_ocr", "query_summary",
    "query_translate", "query_extract", "query_search", "query_system", "query_other",
    "query_file_scan", "query_file_analyze", "query_readonly_report",
]

INTENT_GUIDE = [
    ("add_file", "添加/新增/创建/新建/写入一个文件，例如「添加文件 notes.txt 内容为 ...」"),
    ("delete_file", "删除/移除/清理一个现有文件"),
    ("modify_file", "修改/更新/编辑一个现有文件的内容"),
    ("refactor", "整理/重构/按类型归档/移动多个文件或文件夹"),
    ("query", "查询/搜索/查找本地文件与信息的事实问题，如「搜索 强化学习」「找到红警3的启动程序在哪里」"),
    ("deep_search", "深度分析/对比/综合多个主题或文件，如「对比分析 a 和 b 的差异」"),
    ("vague", "指令模糊、缺操作、缺目标、意图不明，需要向用户澄清"),
    ("file_analysis", "文件深度分析：只读分析项目文件结构、代码质量、依赖关系等"),
    ("file_scan", "文件扫描：只读扫描目录，列出文件、统计信息、检测特殊标记"),
    ("report_generate", "生成报告：基于分析结果生成Markdown/JSON/CSV格式的审查报告"),
    ("filename_query", "按名称搜索文件：查找最近添加/修改/创建的文件名，如「为我找到最近添加的文件名」「最近修改的文件有哪些」"),
]

QUERY_SUBINTENT_GUIDE = [
    ("query_metadata", "查文件信息：文件大小、修改时间、创建时间、属性等元数据"),
    ("query_content", "查文件内容：读取文件的文本内容，如「查看 notes.txt 里写了什么」"),
    ("query_ocr", "OCR识别：识别图片中的文字，如「识别截屏中的文字」「提取图片内容」"),
    ("query_summary", "总结内容：对文件内容进行摘要概括，如「总结这个文件的主要内容」"),
    ("query_translate", "翻译内容：将文件内容翻译成其他语言，如「把这个文件翻译成英文」"),
    ("query_extract", "提取字段：从文件中提取特定信息，如「提取文件中的关键数据」"),
    ("query_search", "搜索信息：在文件库中搜索特定内容，如「搜索强化学习相关文件」"),
    ("query_system", "系统查找：查找系统中的程序或文件，如「找到红警3的启动程序在哪里」"),
    ("query_file_scan", "文件扫描：只读扫描项目文件，如「扫描项目中的所有Python文件」"),
    ("query_file_analyze", "文件分析：深度分析文件内容，如「分析代码复杂度」「检查TODO标记」"),
    ("query_readonly_report", "只读报告：生成只读审查报告，如「生成代码审查报告」"),
]

SYSTEM_PROMPT = (
    "你是 AgentRL 系统的意图路由调度器。请把用户的中文指令分类。\n\n"
    "【一级意图】（11类）：\n"
    + "\n".join(f"  - {v}：{desc}" for v, desc in INTENT_GUIDE) + "\n\n"
    "【QUERY 子意图】（当一级意图为 query 时，需进一步分类）：\n"
    + "\n".join(f"  - {v}：{desc}" for v, desc in QUERY_SUBINTENT_GUIDE) + "\n\n"
    "判断要点：\n"
    "  1. 以用户想「做什么」为准，而不是字面有没有文件扩展名。\n"
    "  2. 「找到/查一下/在哪/搜索」属于 query；「分析/对比/趋势/影响」属于 deep_search。\n"
    "  3. query 是单轮检索即可回答的事实问题；deep_search 需要多轮迭代的分析性任务。\n"
    "  4. 只有明确的增/删/改操作指令才算 add/delete/modify，否则优先 query。\n"
    "  5. 打招呼、闲聊、与文件系统无关的日常话题，一律归为 vague。\n"
    "  6. 当意图为 query 时，必须指定 query_sub_intent 字段。\n"
    "  7. OCR相关：「识别截屏文字」「提取图片内容」属于 query_ocr。\n"
    "  8. 最近截屏：「最近的截屏」「刚才的截图」需先定位文件再OCR。\n"
    "  9. 新增：只读分析/扫描/报告生成属于独立意图，不要误判为query或deep_search。\n"
    "  10. 新增：包含「只读」「不修改」「禁止」等关键词时，必须使用只读路径。\n"
    "  11. 新增：按名称搜索文件（最近添加/修改/创建）属于 filename_query 意图。\n\n"
    "只输出一个 JSON 对象，不要任何解释或 Markdown：\n"
    '  {"intent": "<一级意图>", "query_sub_intent": "<子意图，仅query时填写>", '
    '"target": "<涉及的目标文件/实体>", "slots": {"file_path": "", "image_path": "", "language": ""}, '
    '"confidence": 0到1的小数}'
)


def _extract_json(text: str):
    """宽松提取 JSON：去掉代码围栏，取第一个 { 到最后一个 } 之间的内容。"""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


class LLMIntentClassifier:
    """LLM 意图分类器，支持主端点 -> 本地备端点的容灾链。classify() 全失败返回 None。"""

    def __init__(self):
        self.provider = ""        # 当前生效端点（主或备）的字段，兼容旧调用方
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.timeout = 15
        self.status = "unconfigured"   # ready / fallback / unconfigured
        self.endpoints: list = []
        self._warned = False
        self._config_reason = ""
        self._load_config()

    @property
    def available(self) -> bool:
        return self.status in ("ready", "fallback")

    def _log_status(self, extra: str = ""):
        if self.status == "ready":
            print(f"  [LLM意图] 已启用 provider={self.provider} model={self.model}")
        elif self.status == "fallback":
            print(f"  [LLM意图] 主端点未就绪({extra or self._config_reason})，"
                  f"降级到本地备端点 provider={self.provider} model={self.model}")
        elif not self._warned:
            reason = extra or getattr(self, "_config_reason", "")
            print(f"  [LLM意图] 未启用大模型，回退到规则/Q网络分类。{reason}".rstrip())
            self._warned = True

    # -------------------------------------------------------------- 配置装配
    def _build_endpoint(self, cfg: dict, models_cfg: dict, loader, label: str) -> dict:
        provider = (cfg.get("provider") or "openai")
        model = (cfg.get("model") or "").strip()
        timeout = float(cfg.get("timeout", 15) or 15)

        prov_cfg = (models_cfg.get("providers") or {}).get(provider, {}) or {}
        base_url = (cfg.get("base_url") or prov_cfg.get("base_url", "") or "").rstrip("/")
        auth_env = (cfg.get("auth_env") or prov_cfg.get("auth_env", "") or "").strip()

        api_key = loader.get_secret(auth_env) if auth_env else ""
        if not api_key:
            api_key = prov_cfg.get("api_key", "") or ""
        api_key = api_key.strip()

        reason = ""
        if provider == "local":
            # Ollama 忽略 Authorization，不需要真实密钥
            api_key = "local"
        else:
            if not api_key:
                reason = f"未找到密钥（准备读取环境变量 {auth_env or '(未配置auth_env)'}）"
            elif "*" in api_key or api_key.startswith("your-") or "-your-" in api_key or "your-key-here" in api_key:
                api_key = ""
                reason = "检测到密钥疑似被遮罩（含 * 或占位符），请填入 .env 中不带星号的真实 key"
            elif len(api_key) < 20:
                api_key = ""
                reason = f"密钥过短({len(api_key)}字符)，请核对 .env 中的真实 key"

        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout,
            "reason": reason,
            "ready": bool(provider and model and base_url and api_key),
            "label": label,
            "is_local": provider == "local",
        }

    def _load_config(self):
        try:
            loader = get_config_loader()
            app = loader.load_app_config() or {}
            intent_cfg = (app.get("intent_llm") or {})
            models_cfg = loader.load_models_config() or {}

            endpoints = [self._build_endpoint(intent_cfg, models_cfg, loader, "primary")]
            fb_cfg = intent_cfg.get("fallback")
            if isinstance(fb_cfg, dict) and fb_cfg:
                merged = dict(intent_cfg)
                merged.update({k: v for k, v in fb_cfg.items() if v not in (None, "", {})})
                endpoints.append(self._build_endpoint(merged, models_cfg, loader, "fallback"))
            self.endpoints = endpoints

            primary = endpoints[0]
            if primary["ready"]:
                self.status = "ready"
            elif any(e["label"] == "fallback" and e["ready"] for e in endpoints):
                self.status = "fallback"
            else:
                self.status = "unconfigured"
                self._config_reason = primary.get("reason", "")

            # 当前生效端点：主就绪用主，否则用第一个就绪的备端点
            active = primary if primary["ready"] else next(
                (e for e in endpoints if e["ready"]), primary)
            self.provider = active["provider"]
            self.model = active["model"]
            self.base_url = active["base_url"]
            self.api_key = active["api_key"]
            self.timeout = active["timeout"]

            self._log_status(extra=primary.get("reason", ""))
        except Exception as e:
            self._config_reason = f"配置加载失败: {e}"
            self._log_status(extra=self._config_reason)

    # ------------------------------------------------------------- 主入口
    def classify(self, query: str):
        """按端点链依次尝试，返回 (intent_value, target, confidence, sub_intent, slots)；全失败返回 None。"""
        if not self.available:
            return None

        last_err = None
        for ep in self.endpoints:
            if not ep.get("ready"):
                continue
            try:
                content = self._call_endpoint(query, ep)
            except Exception as e:
                last_err = e
                continue

            data = _extract_json(content)
            if not data or not isinstance(data, dict):
                last_err = ValueError(f"无法解析返回内容: {content[:80]!r}")
                continue

            intent = str(data.get("intent", "")).strip()
            if intent not in VALID_INTENTS:
                last_err = ValueError(f"意图不在允许集合内: {intent!r}")
                continue

            target = str(data.get("target", "")).strip()
            try:
                confidence = float(data.get("confidence", 0.8))
            except (TypeError, ValueError):
                confidence = 0.8
            confidence = max(0.0, min(1.0, confidence))

            sub_intent = str(data.get("query_sub_intent", "")).strip()
            if intent == "query" and sub_intent not in VALID_QUERY_SUBINTENTS:
                sub_intent = "query_other"

            slots = data.get("slots", {})
            if not isinstance(slots, dict):
                slots = {}

            return intent, (target or None), confidence, sub_intent, slots

        if not self._warned:
            self._warned = True
            print(f"  [LLM意图] 所有端点调用/解析失败({last_err})，回退到规则/Q网络分类。")
        return None

    # ------------------------------------------------------------------ HTTP
    def _call_endpoint(self, query: str, ep: dict) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"用户输入：{query}"},
        ]
        if ep["provider"] == "anthropic":
            return self._call_anthropic(messages, ep)
        if ep["provider"] == "gemini":
            return self._call_gemini(messages, ep)
        return self._call_openai_compatible(messages, ep)

    def _call_openai_compatible(self, messages, ep) -> str:
        payload = {
            "model": ep["model"],
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 200,
        }
        if not ep.get("is_local"):
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ep["base_url"] + "/chat/completions", data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ep['api_key']}",
            },
        )
        with urllib.request.urlopen(req, timeout=ep["timeout"]) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, messages, ep) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_text = "".join(m["content"] for m in messages if m["role"] == "user")
        payload = {
            "model": ep["model"],
            "max_tokens": 200,
            "system": system,
            "messages": [{"role": "user", "content": user_text}],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ep["base_url"] + "/messages", data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ep["api_key"],
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=ep["timeout"]) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def _call_gemini(self, messages, ep) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "user", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] == "user"
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200},
        }
        body = json.dumps(payload).encode("utf-8")
        url = (f"{ep['base_url']}/v1beta/models/{ep['model']}:generateContent?"
               f"key={urllib.parse.quote(ep['api_key'])}")
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=ep["timeout"]) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"]