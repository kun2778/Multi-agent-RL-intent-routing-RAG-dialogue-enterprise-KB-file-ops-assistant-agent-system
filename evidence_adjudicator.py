"""跨轮证据充分性裁决模块。

定位：
  - 单轮内部：先 rerank 精选（见 rerank.py / deep_search._iterative_retrieve）
  - 跨轮聚合：本模块对已累积的多轮证据做充分性裁决

裁决信号（聚合去重后计算）：
  - quality    ：证据文档重排得分均值
  - coverage  ：用户查询关键词在证据全文中的覆盖率
  - diversity ：命中文件数（跨轮去重）
  - saturation：最近一轮是否没有带来新文件（检索饱和）

返回值约定：
  - (True/False, reason)：置信裁决，直接采用
  - None：无法判定，交回原有启发式逻辑（evidence_evaluator 的阈值规则）
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from agent_state import AgentState


_STOPWORDS = set(
    ("的 和 与 在 是 有 及 或 把 被 对 从 向 给 让 使 会 能 可 又 也 很 太 还 "
     "只 才 就 都 但 因 为 所 以 然 而 且 如 果 我 们 你 们 他 们 这 那 些 哪 "
     "个 什 么 怎 为 何 么 没 需 要 应 该 必 须 进 行 通 过 方 面 情 况 问 题 "
     "结 果 原 因 之 其 该 此 并 且 请 求 帮 找 分 析").split()
    + ["什么", "怎么", "为什么", "如何", "有没有", "没有", "可以", "需要",
       "应该", "必须", "进行", "通过", "方面", "情况", "问题", "结果", "原因"]
)


def _query_keywords(query: str) -> Set[str]:
    """从查询抽取中英文关键词（CJK 二元组 + 拉丁词），去停用词。"""
    q = (query or "").lower()
    keywords: Set[str] = set()
    for w in re.findall(r"[a-z0-9_]{2,}", q):
        if w not in _STOPWORDS:
            keywords.add(w)
    for run in re.findall(r"[\u4e00-\u9fff]+", q):
        if len(run) == 1:
            if run not in _STOPWORDS:
                keywords.add(run)
            continue
        for i in range(len(run) - 1):
            gram = run[i:i + 2]
            if gram not in _STOPWORDS:
                keywords.add(gram)
    return keywords


class CrossRoundEvidenceAdjudicator:
    """跨轮证据充分性裁决器（只在置信时给出结论）。"""

    def __init__(self,
                 min_quality: float = 0.55,
                 min_diversity: int = 3,
                 min_coverage: float = 0.35,
                 strong_coverage: float = 0.75,
                 strong_quality: float = 0.65,
                 weak_quality: float = 0.45,
                 min_rounds: int = 2):
        self.min_quality = min_quality
        self.min_diversity = min_diversity
        self.min_coverage = min_coverage
        self.strong_coverage = strong_coverage
        self.strong_quality = strong_quality
        self.weak_quality = weak_quality
        self.min_rounds = min_rounds

    # ------------------------------------------------------------------ 入口
    def adjudicate(self, state: AgentState) -> Optional[Tuple[bool, str]]:
        """跨轮裁决：返回 (是否充分, 原因)；无法判定返回 None。"""
        rounds = len(state.evidence)
        if rounds < self.min_rounds:
            return None

        docs = self._collect_documents(state)
        if not docs:
            return None

        query = state.dialog_history[-1].user_query if state.dialog_history else ""
        keywords = _query_keywords(query)
        if not keywords:
            return None

        text_union = " ".join(
            str(d.get("content", "")).lower() for d in docs
        )
        matched = sum(1 for k in keywords if k in text_union)
        coverage = matched / len(keywords)

        quality = self._quality(docs)
        diversity = self._diversity(docs)
        saturated = self._saturated(state)

        # ---- 置信「充分」 ----
        if (diversity >= self.min_diversity
                and quality >= self.min_quality
                and coverage >= self.min_coverage):
            return True, (
                f"[跨轮裁决] 证据充分：{rounds} 轮聚合 {len(docs)} 条证据，"
                f"质量 {quality:.2f}，查询覆盖 {coverage:.0%}，"
                f"命中 {diversity} 个不同文件"
            )
        if coverage >= self.strong_coverage and quality >= self.strong_quality \
                and diversity >= 2:
            return True, (
                f"[跨轮裁决] 证据充分：查询覆盖率达 {coverage:.0%}，"
                f"质量 {quality:.2f}，命中 {diversity} 个文件"
            )
        if saturated and quality >= self.min_quality and diversity >= 2 \
                and coverage >= self.min_coverage:
            return True, (
                f"[跨轮裁决] 证据充分：检索已饱和（最近一轮无新文件），"
                f"质量 {quality:.2f}，覆盖 {coverage:.0%}"
            )

        # ---- 置信「不充分」 ----
        if diversity <= 1:
            return False, (
                f"[跨轮裁决] 证据不充分：{rounds} 轮仅命中 {diversity} 个文件，"
                f"需要换角度继续检索"
            )
        if rounds >= 3 and saturated and quality < self.weak_quality:
            return False, (
                f"[跨轮裁决] 证据不充分：最近一轮无新增文件（检索饱和）"
                f"且质量偏低 {quality:.2f}，需调整检索方向"
            )

        # ---- 无法置信判定：交回原启发式 ----
        return None

    # ------------------------------------------------------------------ 信号
    def _collect_documents(self, state: AgentState) -> List[Dict]:
        """聚合跨轮文档并按 (path, 内容前缀) 去重。"""
        seen: Set[Tuple[str, str]] = set()
        docs: List[Dict] = []
        for ev in state.evidence:
            for doc in ev.documents:
                if not isinstance(doc, dict):
                    continue
                path = str(doc.get("path", "") or doc.get("source", ""))
                prefix = str(doc.get("content", ""))[:80]
                key = (path, prefix)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
        return docs

    def _quality(self, docs: List[Dict]) -> float:
        """证据质量：有效文档（score>=0.3）的平均得分。"""
        scores = []
        for d in docs:
            try:
                s = float(d.get("score", 0.0))
            except (TypeError, ValueError):
                s = 0.0
            if s >= 0.3:
                scores.append(min(s, 1.0))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _diversity(self, docs: List[Dict]) -> int:
        files = set()
        for d in docs:
            path = d.get("path") or d.get("source")
            if path:
                files.add(path)
        return len(files)

    def _saturated(self, state: AgentState) -> bool:
        """最近一轮是否没有带来新文件（跨轮检索饱和信号）。"""
        if len(state.evidence) < 2:
            return False

        def files_of(ev) -> Set[str]:
            return {
                str(d.get("path") or d.get("source") or "")
                for d in ev.documents
                if isinstance(d, dict) and (d.get("path") or d.get("source"))
            }

        last = files_of(state.evidence[-1])
        if not last:
            return True
        earlier: Set[str] = set()
        for ev in state.evidence[:-1]:
            earlier |= files_of(ev)
        return last.issubset(earlier)
