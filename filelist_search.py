"""FILELIST_SEARCH 意图执行器：读取文件列表 + 云 LLM ReAct 迭代扫描。

职责边界
--------
- **只读**：只做列目录 / 递归遍历 / 文件名与元数据匹配 / 注册表只读查询，
  绝不写盘、不删除、不改名。
- **输入**：一个根路径 + 用户条件（关键词、扩展名、类型提示、时间下限），
  例如「为我找出 C:\\Steam 路径下的 steam 注册表文件」。
- **ReAct 扫描**：每一轮由云端 LLM 决定下一步动作
  （list_dir / scan / match / registry / answer），
  观察结果回灌提示词，直到 LLM 给出 answer、轮次用尽或时间预算耗尽。
- **无密钥降级**：`deep_search.ReActLLMClient` 读取的是同一套端点链，
  key 未配置 / 端点全挂时 `available=False`，这里退回确定性的规则扫描计划，
  主流程不中断、不抛异常。
"""

from __future__ import annotations

import os
import re
import fnmatch
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from agent_state import AgentState
from deep_search import ReActLLMClient, _extract_json

try:
    import winreg
except ImportError:            # 非 Windows 平台：注册表工具直接不可用
    winreg = None


# ================================================================
# ReAct 提示词
# ================================================================
REACT_SYSTEM_PROMPT = """\
你是 AgentRL 的文件列表扫描器。你正在执行 ReAct（思考-行动-观察）循环，
目标是：在用户指定的根路径下，按用户条件找出符合条件的文件清单。

## 约束（必须遵守）
1. 只读：只列目录、遍历、匹配文件名与元数据，禁止任何写入/删除/改名。
2. 预算有限：目录深度与访问条目都有上限，先浅后深，不要一次扫全盘。
3. 每一轮只能调用一个工具，不要重复上一轮已经做过的动作。
4. 找到足够匹配、或轮次将尽时，输出 action=answer。

## 可用工具
1) list_dir —— 列出某个目录的直接子项
   {{"thought": "...", "action": "list_dir", "params": {{"path": "C:\\\\Steam", "limit": 100}}}}
2) scan —— 从某个目录向下遍历，按条件匹配文件（会回灌命中的文件清单）
   {{"thought": "...", "action": "scan", "params": {{"path": "C:\\\\Steam", "max_depth": 6, "limit": 200}}}}
3) match —— 对已发现的文件做二次过滤（glob 或正则，作用于文件名）
   {{"thought": "...", "action": "match", "params": {{"pattern": "*steam*.reg"}}}}
4) registry —— 只读查询 Windows 注册表，按关键词找键/值/路径
   （用于「注册表文件」「安装位置」这类请求）
   {{"thought": "...", "action": "registry", "params": {{"keyword": "steam"}}}}
5) answer —— 输出最终文件清单
   {{"thought": "...", "action": "answer", "params": {{"summary": "基于以上扫描给出结论"}}}}

## 输出格式
只输出一个 JSON 对象，不要 Markdown 围栏，不要解释。
首次行动前先看「根路径观察」，据此决定先扫哪一层。
"""


# ================================================================
# 关键词 → 扩展名类型提示（文件名里没有该词时，用扩展名族兜底）
# ================================================================
TYPE_HINT_EXTS: Dict[str, Tuple[str, ...]] = {
    "注册表": (".reg",),
    "registry": (".reg",),
    "日志": (".log", ".txt"),
    "log": (".log",),
    "配置": (".yaml", ".yml", ".json", ".ini", ".cfg", ".conf", ".toml"),
    "config": (".yaml", ".yml", ".json", ".ini", ".cfg", ".conf", ".toml"),
    "快捷方式": (".lnk",),
    "shortcut": (".lnk",),
    "脚本": (".py", ".ps1", ".bat", ".cmd", ".sh", ".js"),
    "script": (".py", ".ps1", ".bat", ".cmd", ".sh", ".js"),
    "图片": (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"),
    "image": (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"),
    "压缩包": (".zip", ".7z", ".rar", ".tar", ".gz"),
    "archive": (".zip", ".7z", ".rar", ".tar", ".gz"),
    "可执行": (".exe", ".dll", ".msi"),
    "program": (".exe", ".dll", ".msi"),
}

#: 无点写法也算扩展名的常见后缀（「找出 E:\\backup 下的 txt 文件」）
COMMON_EXTS = {
    "txt", "log", "reg", "py", "json", "yaml", "yml", "csv", "md", "xml",
    "ini", "cfg", "conf", "toml", "pdf", "doc", "docx", "xls", "xlsx",
    "ppt", "pptx", "png", "jpg", "jpeg", "gif", "bmp", "webp", "zip",
    "7z", "rar", "tar", "gz", "exe", "dll", "bat", "cmd", "ps1", "sh",
    "js", "ts", "html", "css", "mp3", "mp4", "mkv", "avi", "iso", "msi",
    "lnk", "db", "sqlite", "bak", "tmp", "dat",
}

#: 中文条件词里必须剥掉的通用名词/动作词（长的排前面，循环剥到剥不动为止）
_GENERIC_SUFFIXES = (
    "的相关文件", "所有文件", "相关文件", "路径下的", "目录下的", "文件夹下的",
    "的文件", "文件夹", "文件", "文档", "程序", "目录", "路径", "清单",
    "列表", "内容", "相关", "所有", "全部", "下面", "里面", "找出", "列出",
    "看看", "查找", "搜索", "查询", "列举", "遍历", "扫描", "文件名",
    "列出来", "有哪些", "有什么", "是什么", "有没有", "下的", "里的", "的",
)

#: 剥完仍然是「通用词」的 → 不是真关键词，丢弃
_GENERIC_WORDS = {
    "路径", "目录", "文件夹", "文件", "文档", "程序", "清单", "列表",
    "内容", "所有", "全部", "里面", "下面", "相关", "位置", "那里",
    "这里", "哪些", "什么", "东西", "注册表文件",
}

#: 段首的客气动词/介词（先剥前缀，再剥后缀，剩下的才是真关键词）
_LEADING_WORDS = (
    "为我", "请你", "麻烦", "帮我", "帮忙", "给我", "列出", "找出",
    "看看", "扫描", "搜索", "查找", "查询", "列举", "遍历", "请", "把", "在",
)

_STOPWORDS = {
    "这个", "那个", "这些", "那些", "一个", "一下", "帮我", "请你", "请问",
    "为我", "给我", "可以", "哪些", "什么", "怎么", "有没有", "是否",
    "以及", "还有", "其中", "下面", "里面", "under", "list", "files",
    "file", "find", "search", "path", "dir", "directory", "the", "and",
}

#: 遍历时直接跳过的系统/依赖目录（根路径是 C:\\ 时靠它保住预算）
PRUNE_DIRS = {
    "$recycle.bin", "system volume information", "windows", "winsxs",
    "recovery", "node_modules", "__pycache__", ".git", ".svn",
    ".cache", ".gradle", ".m2", "$windows.~bt",
}


@dataclass
class FileListSpec:
    """一次文件列表检索的结构化条件。"""
    root: str = ""
    keywords: List[str] = field(default_factory=list)
    exts: List[str] = field(default_factory=list)
    registry: bool = False
    mtime_after: Optional[float] = None
    max_depth: int = 8
    limit: int = 200
    query: str = ""

    def describe(self) -> str:
        parts = []
        if self.keywords:
            parts.append("关键词=" + "/".join(self.keywords))
        if self.exts:
            parts.append("扩展名=" + ",".join(self.exts))
        if self.registry:
            parts.append("注册表辅助=是")
        if self.mtime_after:
            parts.append("修改时间≥" + datetime.fromtimestamp(self.mtime_after).strftime("%Y-%m-%d"))
        return "  ".join(parts) or "（无附加条件，列出全部文件）"


class FileListSearchHandler:
    """文件列表检索：确定性工具 + 云 LLM ReAct 调度。"""

    ACTIONS = ("list_dir", "scan", "match", "registry", "answer")

    def __init__(self, max_rounds: int = 5, entry_budget: int = 100000,
                 top_k: int = 50, max_matches: int = 5000):
        self.max_rounds = max_rounds
        self.entry_budget = entry_budget
        self.top_k = top_k
        self.max_matches = max_matches
        self.llm = ReActLLMClient()
        self.trace: List[Dict[str, Any]] = []
        self.observations: List[str] = []
        self.matches: Dict[str, Dict[str, Any]] = {}
        self.suggested_dirs: List[str] = []
        self.registry_hits: List[str] = []
        self.entries_visited = 0
        self.dirs_visited = 0
        self.budget_exhausted = False
        self.match_cap_hit = False
        self.found = False
        self.llm_used = False

    # ------------------------------------------------------------ 查询解析
    # 盘符路径：字符集里排除中日韩与中文标点 —— 「C:\Steam路径下的…」必须截在 Steam
    _DRIVE_PATH_RE = re.compile(
        r'[A-Za-z]:[\\/][A-Za-z0-9_\-.\\/ \'()&\[\]]*')
    _ABS_PATH_RE = re.compile(r'(?<![\w.])[~/][^\s，。；;、！？!?"\'“”‘’（）()<>|*?]+')

    @classmethod
    def _extract_root(cls, query: str, slots: Optional[Dict[str, Any]] = None) -> str:
        """从查询里取根路径：存在的目录优先；给过但不存在的路径也要带回去报错。"""
        q = query or ""
        candidates: List[str] = []
        for raw in cls._DRIVE_PATH_RE.findall(q):
            cand = raw.rstrip(" \\,'\"")
            if cand:
                candidates.append(cand)
        for raw in cls._ABS_PATH_RE.findall(q):
            cand = os.path.abspath(os.path.expanduser(raw.rstrip("\\/")))
            if cand:
                candidates.append(cand)
        for cand in candidates:
            if os.path.isdir(cand):
                return cand
            parts = cand.split(" ")
            for i in range(len(parts) - 1, 0, -1):
                sub = " ".join(parts[:i]).rstrip(" \\,'\"")
                if sub and os.path.isdir(sub):
                    return sub
        slot_path = str((slots or {}).get("file_path") or "")
        if slot_path and os.path.isdir(slot_path):
            return slot_path
        return candidates[0] if candidates else ""

    @classmethod
    def _extract_keywords(cls, query: str, root: str) -> List[str]:
        """剥掉路径与通用词，剩下 文件名关键词 + 中文条件词。"""
        text = query or ""
        if root:
            text = text.replace(root, " ")
        text = text.replace("\\", " ")
        kws: List[str] = []

        for m in re.finditer(r"\.([A-Za-z0-9]{1,6})\b", text):
            kws.append("." + m.group(1).lower())

        for m in re.finditer(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", text):
            low = m.group(0).lower()
            if m.start() > 0 and text[m.start() - 1] == ".":
                continue                      # 已作为扩展名收过，不再当普通关键词
            if low in COMMON_EXTS:
                kws.append("." + low)         # 「txt 文件」这种无点写法
                continue
            if low in _STOPWORDS or low in ("com", "exe", "org"):
                continue
            kws.append(low)

        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            # 「目录下的日志文件」→ 按 下/里/的 切段，逐段取真关键词，
            # 通用名词段（目录/所有/文件…）剥完是空串，直接丢弃。
            for part in re.split(r"路径下|目录下|文件夹下|里面的|里头|里面|下的|里的|的", run):
                word = part
                for _ in range(4):
                    for lead in _LEADING_WORDS:
                        if word.startswith(lead) and len(word) > len(lead):
                            word = word[len(lead):]
                            break
                    else:
                        break
                for _ in range(5):
                    stripped = False
                    for suf in _GENERIC_SUFFIXES:
                        if word.endswith(suf) and len(word) >= len(suf):
                            word = word[: -len(suf)]
                            stripped = True
                            break
                    if not stripped:
                        break
                if len(word) >= 2 and word not in _STOPWORDS and word not in _GENERIC_WORDS:
                    kws.append(word)

        seen, out = set(), []
        for k in kws:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    @staticmethod
    def _extract_mtime_after(query: str) -> Optional[float]:
        q = (query or "").lower()
        now = datetime.now()
        if "今天" in q:
            return now.replace(hour=0, minute=0, second=0).timestamp()
        if "昨天" in q:
            return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0).timestamp()
        if "本周" in q or "这周" in q:
            return (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0).timestamp()
        m = re.search(r"最近(\d+)\s*(天|小时|分钟)", q)
        if m:
            n = int(m.group(1))
            unit = {"天": 86400, "小时": 3600, "分钟": 60}[m.group(2)]
            return time.time() - n * unit
        return None

    def parse_spec(self, query: str, slots: Optional[Dict[str, Any]] = None) -> FileListSpec:
        """把自然语言查询解析成 FileListSpec（root 为空表示需要澄清）。"""
        slots = slots or {}
        root = self._extract_root(query, slots)
        keywords = self._extract_keywords(query, root)
        exts = [k for k in keywords if k.startswith(".") and len(k) > 1]
        plain = [k for k in keywords if not k.startswith(".")]
        registry = any(k in ("注册表", "registry") for k in plain)
        return FileListSpec(
            root=root,
            keywords=plain,
            exts=exts,
            registry=registry,
            mtime_after=self._extract_mtime_after(query),
            query=query or "",
        )

    @staticmethod
    def clarification_for(query: str) -> List[str]:
        """缺根路径时生成的澄清问题（router 直接消费）。"""
        return ["没有识别到要检索的根路径，请补充路径，例如："
                "「为我找出 C:\\Steam 路径下的 steam 注册表文件」"]

    # ------------------------------------------------------------ 执行入口
    def execute(self, state: AgentState, spec: FileListSpec) -> str:
        """ReAct 扫描一轮轮推进，返回最终的文件清单文本。"""
        self._reset()
        t0 = time.time()

        if not spec.root or not os.path.isdir(spec.root):
            return self._format_report(spec, error=f"根路径不存在或不是目录: {spec.root!r}")

        print(f"\n  [文件列表 ReAct] 根路径: {spec.root}  条件: {spec.describe()}")
        if self.llm.available:
            self.llm_used = True
            print(f"  [LLM] 可用（{self.llm.status}: {self.llm.endpoint}）→ 云端 ReAct 扫描")
        else:
            print(f"  [LLM] 不可用（{self.llm.reason or '未配置'}）→ 降级为规则扫描计划")

        seed = self._tool_list_dir(spec.root, limit=30)
        self.observations.append(seed)

        plan = self._rule_plan(spec)
        for step in range(1, self.max_rounds + 1):
            if state.elapsed_time > state.retrieval_budget:
                self.trace.append({"step": step, "thought": "时间预算耗尽",
                                   "action": "stop", "detail": "budget"})
                break

            decision = None
            if self.llm_used:
                decision = self._llm_step(spec, state, step)
            if decision is None:
                decision = plan.pop(0) if plan else ("answer", "规则计划已执行完毕", {})

            action, thought, params = decision
            if action == "answer":
                self.trace.append({"step": step, "thought": thought,
                                   "action": "answer", "detail": (params or {}).get("summary", "")[:200]})
                print(f"  [Thought] {thought[:120]}")
                print("  [Action] → answer")
                break

            print(f"  [Thought] {thought[:120]}")
            print(f"  [Action] → {action}: {self._brief(params)}")
            obs = self._dispatch(action, params, spec, state)
            self.observations.append(obs)
            self.trace.append({"step": step, "thought": thought,
                               "action": action, "detail": obs[:200]})

            if self.budget_exhausted:
                self.trace.append({"step": step, "thought": "条目预算耗尽",
                                   "action": "stop", "detail": "entry_budget"})
                break

        if not self.llm_used and (not self.trace or self.trace[-1]["action"] != "answer"):
            self.trace.append({"step": len(self.trace) + 1, "thought": "规则扫描计划执行完毕",
                               "action": "answer", "detail": "fallback"})

        self.found = bool(self.matches)
        elapsed = time.time() - t0
        print(f"  [扫描] 目录 {self.dirs_visited} 个 / 条目 {self.entries_visited} 个 / "
              f"命中 {len(self.matches)} 个 / 用时 {elapsed:.2f}s")
        return self._format_report(spec, elapsed=elapsed)

    def _reset(self):
        self.trace = []
        self.observations = []
        self.matches = {}
        self.suggested_dirs = []
        self.registry_hits = []
        self.entries_visited = 0
        self.dirs_visited = 0
        self.budget_exhausted = False
        self.match_cap_hit = False
        self.found = False
        self.llm_used = False

    # ------------------------------------------------------------ 决策
    def _rule_plan(self, spec: FileListSpec) -> List[Tuple[str, str, Dict[str, Any]]]:
        """LLM 缺席时的确定性扫描计划（先扫、必要时补注册表、收口）。"""
        plan: List[Tuple[str, str, Dict[str, Any]]] = [
            ("scan", "先在根路径下做一次带条件的深度扫描",
             {"path": spec.root, "max_depth": spec.max_depth, "limit": spec.limit}),
        ]
        if spec.registry:
            kw = spec.keywords[0] if spec.keywords else ""
            plan.append(("registry", "条件里带注册表语义，补一次注册表只读查询",
                         {"keyword": kw}))
        if not plan or plan[-1][0] != "answer":
            plan.append(("answer", "扫描完成，汇总命中的文件清单", {}))
        return plan

    def _llm_step(self, spec: FileListSpec, state: AgentState,
                  step: int) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """向云端 LLM 要一步 ReAct 决策；调用/解析失败返回 None（走规则计划）。"""
        user = self._build_user_prompt(spec, state, step)
        raw = self.llm.chat(REACT_SYSTEM_PROMPT, user)
        if not raw:
            return None
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None
        action = str(data.get("action", "")).strip().lower()
        if action not in self.ACTIONS:
            return None
        params = data.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        thought = str(data.get("thought", ""))[:400]
        return action, thought, params

    def _build_user_prompt(self, spec: FileListSpec, state: AgentState, step: int) -> str:
        recent = self.observations[-3:]
        matched_preview = "\n".join(
            f"  - {m['path']}" for m in self._sorted_matches()[:10]
        ) or "  （暂无命中）"
        return (
            f"用户请求: {spec.query}\n"
            f"根路径: {spec.root}\n"
            f"已提取条件: {spec.describe()}\n"
            f"当前第 {step}/{self.max_rounds} 轮；"
            f"已访问目录 {self.dirs_visited} 个 / 条目 {self.entries_visited} 个 "
            f"（上限 {self.entry_budget}）\n\n"
            f"=== 最近观察 ===\n" + "\n".join(recent) + "\n\n"
            f"=== 当前命中（{len(self.matches)} 个） ===\n{matched_preview}\n\n"
            "请输出下一步行动的 JSON。"
        )

    # ------------------------------------------------------------ 工具分发
    @staticmethod
    def _brief(params: Dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in list((params or {}).items())[:4])

    def _dispatch(self, action: str, params: Dict[str, Any], spec: FileListSpec,
                  state: AgentState) -> str:
        params = params or {}
        try:
            if action == "list_dir":
                return self._tool_list_dir(str(params.get("path") or spec.root),
                                           int(params.get("limit") or 100))
            if action == "scan":
                return self._tool_scan(str(params.get("path") or spec.root), spec,
                                       max_depth=int(params.get("max_depth") or spec.max_depth),
                                       limit=int(params.get("limit") or spec.limit),
                                       pattern=str(params.get("pattern") or ""))
            if action == "match":
                return self._tool_match(str(params.get("pattern") or ""),
                                        int(params.get("limit") or spec.limit))
            if action == "registry":
                kw = str(params.get("keyword") or (spec.keywords[0] if spec.keywords else ""))
                return self._tool_registry(kw, spec)
        except Exception as exc:            # 工具异常不允许打断主流程
            return f"[工具异常] {action}: {exc}"
        return f"[未知工具] {action}，可用工具: {'/'.join(self.ACTIONS)}"

    # ------------------------------------------------------------ 只读工具
    def _tool_list_dir(self, path: str, limit: int = 100) -> str:
        path = os.path.abspath(path) if path else path
        if not os.path.exists(path):
            return f"路径不存在: {path}"
        if os.path.isfile(path):
            st = os.stat(path)
            return (f"这是一个文件: {path}  "
                    f"({self._human_size(st.st_size)}, {self._fmt_mtime(st.st_mtime)})")

        self.dirs_visited += 1
        dirs, files = [], []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if len(dirs) + len(files) >= limit * 3:
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(entry.name)
                        else:
                            st = entry.stat(follow_symlinks=False)
                            files.append((entry.name, st.st_size, st.st_mtime))
                    except OSError:
                        continue
        except OSError as exc:
            return f"无法读取目录 {path}: {exc}"

        lines = [f"目录 {path}：{len(dirs)} 个子目录 / {len(files)} 个文件（展示前 {limit} 项）"]
        for d in sorted(dirs)[:limit]:
            lines.append(f"  [D] {d}/")
        for name, size, mt in sorted(files)[:limit]:
            lines.append(f"  [F] {name}  {self._human_size(size)}  {self._fmt_mtime(mt)}")
        if len(files) > limit or len(dirs) > limit:
            lines.append("  ...（更多条目已截断，可用 scan 深入）")
        return "\n".join(lines)

    def _tool_scan(self, root: str, spec: FileListSpec, max_depth: int = 8,
                   limit: int = 200, pattern: str = "") -> str:
        root = os.path.abspath(root) if root else spec.root
        if not os.path.isdir(root):
            return f"路径不存在或不是目录: {root}"

        max_depth = max(1, min(int(max_depth or spec.max_depth), 16))
        limit = max(1, min(int(limit or spec.limit), 2000))
        before = len(self.matches)
        stack: List[Tuple[str, int]] = [(root, 0)]
        scanned_files = 0

        while stack and not self.budget_exhausted:
            cur, depth = stack.pop()
            self.dirs_visited += 1
            try:
                with os.scandir(cur) as it:
                    entries = list(it)
            except OSError:
                continue
            for entry in entries:
                if self.entries_visited >= self.entry_budget:
                    self.budget_exhausted = True
                    break
                self.entries_visited += 1
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    if depth < max_depth and entry.name.lower() not in PRUNE_DIRS:
                        stack.append((entry.path, depth + 1))
                    continue
                scanned_files += 1
                if pattern and not self._match_pattern(entry.name, pattern):
                    continue
                info = self._score_file(entry.path, spec)
                if info is None:
                    continue
                if len(self.matches) >= self.max_matches:
                    self.match_cap_hit = True
                    continue
                self.matches[entry.path] = info

        new_hits = len(self.matches) - before
        return (f"scan({root}, depth≤{max_depth}) 完成：访问条目 {self.entries_visited}，"
                f"本轮扫描文件 {scanned_files}，新增命中 {new_hits}，累计命中 {len(self.matches)}")

    def _tool_match(self, pattern: str, limit: int = 200) -> str:
        if not pattern:
            return "match 需要 pattern 参数（glob 或正则，作用于文件名）"
        keep = {p: m for p, m in self.matches.items()
                if self._match_pattern(os.path.basename(p), pattern)}
        removed = len(self.matches) - len(keep)
        self.matches = keep
        return (f"match({pattern}) 完成：保留 {len(self.matches)} 个，"
                f"过滤掉 {removed} 个（展示上限 {min(limit, self.top_k)}）")

    def _tool_registry(self, keyword: str, spec: FileListSpec) -> str:
        """只读查询注册表：按关键词匹配键名/值名/字符串值，附带可用路径建议。"""
        if winreg is None:
            return "当前平台不支持注册表查询（winreg 不可用）"
        kw = (keyword or "").lower()
        if not kw:
            return "registry 需要 keyword 参数"

        roots = [
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Valve\Steam"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]
        hits: List[str] = []
        for hive, sub in roots:
            if len(hits) >= 40:
                break
            try:
                with winreg.OpenKey(hive, sub) as key:
                    self._walk_registry(key, f"{self._hive_name(hive)}\\{sub}", kw, hits, depth=2)
            except OSError:
                continue

        self.registry_hits = hits
        lines = [f"注册表查询「{keyword}」：{len(hits)} 条命中"]
        lines.extend("  " + h for h in hits[:40])
        if not hits:
            lines.append("  （无命中：可改用文件名扫描，或换更精确的关键词）")
        return "\n".join(lines)

    @staticmethod
    def _hive_name(hive) -> str:
        if winreg is None:
            return "?"
        for attr, short in (("HKEY_LOCAL_MACHINE", "HKLM"),
                            ("HKEY_CURRENT_USER", "HKCU"),
                            ("HKEY_CLASSES_ROOT", "HKCR"),
                            ("HKEY_USERS", "HKU")):
            if hive == getattr(winreg, attr, None):
                return short
        return str(hive)

    def _walk_registry(self, key, path: str, kw: str, out: List[str], depth: int):
        if depth < 0 or len(out) >= 40:
            return
        try:
            subkeys = [winreg.EnumKey(key, i) for i in range(winreg.QueryInfoKey(key)[0])]
        except OSError:
            subkeys = []
        for name in subkeys:
            if len(out) >= 40:
                return
            joined = f"{path}\\{name}"
            if kw in name.lower():
                out.append(f"键 {joined}")
            try:
                with winreg.OpenKey(key, name) as sub:
                    for i in range(winreg.QueryInfoKey(sub)[1]):
                        try:
                            vname, value, _ = winreg.EnumValue(sub, i)
                        except OSError:
                            break
                        text = str(value)
                        if kw in vname.lower() or kw in text.lower():
                            out.append(f"值 {joined} :: {vname} = {text[:160]}")
                            self._note_registry_path(text)
                    self._walk_registry(sub, joined, kw, out, depth - 1)
            except OSError:
                continue

    def _note_registry_path(self, value: str):
        for cand in re.findall(r'[A-Za-z]:\\[^\s"\'|]+', value):
            cand = cand.rstrip("\\/,")
            if os.path.isdir(cand):
                if cand not in self.suggested_dirs:
                    self.suggested_dirs.append(cand)
            elif os.path.isfile(cand) and cand not in self.matches:
                info = self._score_file(cand, None)
                if info is not None:
                    self.matches[cand] = info

    # ------------------------------------------------------------ 匹配
    @staticmethod
    def _match_pattern(name: str, pattern: str) -> bool:
        if not pattern:
            return True
        low = (name or "").lower()
        pat = pattern.lower().strip()
        if any(c in pat for c in "*?["):
            return fnmatch.fnmatch(low, pat)
        try:
            return re.search(pat, low) is not None
        except re.error:
            return pat in low

    def _score_file(self, path: str, spec: Optional[FileListSpec]) -> Optional[Dict[str, Any]]:
        """返回该文件的命中信息（None 表示不匹配）。"""
        try:
            st = os.stat(path)
        except OSError:
            return None
        if spec and spec.mtime_after and st.st_mtime < spec.mtime_after:
            return None

        name = os.path.basename(path).lower()
        ext = os.path.splitext(name)[1]
        keywords = list(spec.keywords) if spec else []
        exts = list(spec.exts) if spec else []

        if exts and ext not in exts:
            return None

        name_hits = [k for k in keywords if k in name]
        type_hits = [k for k in keywords if ext in TYPE_HINT_EXTS.get(k, ())]
        # 路径命中只看根路径**以下**的相对路径：根目录名本身不含关键词，
        # 否则「C:\Steam 下的 steam 文件」会让整棵树全部命中。
        rel_hits: List[str] = []
        if not name_hits and not type_hits and spec and spec.root:
            try:
                rel = os.path.relpath(path, spec.root).lower()
            except ValueError:
                rel = ""
            rel_hits = [k for k in keywords if k in rel]
        if keywords and not (name_hits or type_hits or rel_hits):
            return None

        score = 1.0 + len(name_hits) * 3.0 + len(type_hits) * 2.0 + len(rel_hits)
        return {
            "path": path,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "score": score,
            "name_hits": name_hits,
            "type_hits": type_hits,
        }

    def _sorted_matches(self) -> List[Dict[str, Any]]:
        return sorted(self.matches.values(),
                      key=lambda m: (-m["score"], -m["mtime"], m["path"]))

    # ------------------------------------------------------------ 报告
    @staticmethod
    def _human_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
            value /= 1024
        return f"{value:.1f}TB"

    @staticmethod
    def _fmt_mtime(ts: float) -> str:
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            return "-"

    def _format_report(self, spec: FileListSpec, elapsed: float = 0.0,
                       error: str = "") -> str:
        lines = ["  [文件列表检索结果]", "  模式: 只读（ReAct 扫描）"]
        lines.append(f"  根路径: {spec.root or '（未指定）'}")
        lines.append(f"  条件: {spec.describe()}")
        if error:
            lines.append(f"  {error}")
            lines.append("  提示: 请检查路径是否正确，或补充更具体的关键词/扩展名。")
            return "\n".join(lines)

        hits = self._sorted_matches()
        lines.append(f"  匹配 {len(hits)} 个文件"
                     f"（扫描目录 {self.dirs_visited} 个 / 访问条目 {self.entries_visited} 个"
                     f" / 用时 {elapsed:.2f}s）")
        lines.append("")
        if not hits:
            lines.append("  未找到匹配文件。")
            lines.append("  建议: 放宽关键词、补上扩展名（如 .reg/.log），或换一个根路径。")
        else:
            lines.append("  文件清单:")
            for i, m in enumerate(hits[: self.top_k], 1):
                tag = ""
                if m.get("name_hits") or m.get("type_hits"):
                    tag = "  [" + "/".join(m.get("name_hits") or m.get("type_hits")) + "]"
                lines.append(f"    {i:>2}. {m['path']}  "
                             f"{self._human_size(m['size'])}  "
                             f"{self._fmt_mtime(m['mtime'])}{tag}")
            if len(hits) > self.top_k:
                lines.append(f"    ...（共 {len(hits)} 个，仅展示前 {self.top_k} 个）")

        if self.suggested_dirs:
            lines.append("")
            lines.append("  注册表给出的候选目录:")
            for d in self.suggested_dirs[:5]:
                lines.append(f"    - {d}")

        if self.budget_exhausted:
            lines.append("")
            lines.append(f"  注意: 已达到条目预算 {self.entry_budget}，扫描提前结束。")
        if self.match_cap_hit:
            lines.append("")
            lines.append(f"  注意: 命中数已达上限 {self.max_matches}，可用更严格的条件过滤。")

        if self.trace:
            lines.append("")
            lines.append("  ReAct 扫描轨迹:")
            for t in self.trace:
                lines.append(f"    第{t.get('step')}轮 [{t.get('action')}] "
                             f"{(t.get('thought') or '')[:80]}")
            if not self.llm_used:
                lines.append("    （云端 LLM 未启用，本轮由规则扫描计划驱动）")
        return "\n".join(lines)

    # ------------------------------------------------------------ 兼容入口
    def run(self, state: AgentState, query: str,
            slots: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """便捷入口：解析 + 执行，返回 (是否找到, 报告文本)。"""
        spec = self.parse_spec(query, slots if slots is not None else state.slots)
        if not spec.root:
            return False, "\n".join(self.clarification_for(query))
        text = self.execute(state, spec)
        return self.found, text
