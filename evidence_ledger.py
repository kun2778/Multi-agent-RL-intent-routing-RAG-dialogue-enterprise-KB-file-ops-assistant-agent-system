# -*- coding: utf-8 -*-
"""证据台账：ReAct 深度检索的结构化证据记账（方面覆盖 + 主张置信度）。

本模块是仓库内唯一副本（duplicate_gate 跨文件检测），只做纯记账：
不实现文本探针/分词/矛盾词对检测（那些逻辑留在 deep_search.py 与
evidence_evaluator.py），台账只消费调用方传入的结构化结果。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re


@dataclass
class SearchAspect:
    name: str
    covered: bool = False
    required: bool = True


@dataclass
class Claim:
    text: str
    sources: List[str] = field(default_factory=list)  # 独立来源路径（去重）
    score: float = 0.0                                 # 各来源中的最高分
    aspect: str = ""
    confidence: float = 0.0


class EvidenceLedger:
    aspects: List[SearchAspect]
    claims: List[Claim]

    def __init__(self) -> None:
        self.aspects = []
        self.claims = []

    def set_aspects(self, names: List[str], required: Optional[List[str]] = None) -> None:
        """登记检索方面；已存在的 aspect 保留其 covered 状态不覆盖。"""
        # 去重且保序；新增 aspect：required=None 时全部 required=True，
        # 否则 required = (name in required 列表)；新增均为 covered=False。
        existing = {a.name.strip().lower() for a in self.aspects}
        for raw in names:
            name = raw.strip()
            if not name:
                continue
            key = name.lower()
            if key in existing:
                continue
            is_required = True if required is None else (name in required)
            self.aspects.append(SearchAspect(name=name, covered=False, required=is_required))
            existing.add(key)

    def add_claim(self, text: str, source: str, score: float, aspect: str = "") -> None:
        """登记/合并一条主张。"""
        # text 或 source 为空白 → 直接返回。
        if not text or not text.strip() or not source or not source.strip():
            return
        # 归一化合并键：re.sub(r"\s+", "", text).strip().lower()
        key = re.sub(r"\s+", "", text).strip().lower()
        target: Optional[Claim] = None
        for c in self.claims:
            if re.sub(r"\s+", "", c.text).strip().lower() == key:
                target = c
                break
        if target is None:
            # 否则新建 Claim(text=text, sources=[source], score=score)。
            target = Claim(text=text, sources=[source], score=score)
            self.claims.append(target)
        else:
            # 同键已存在 → score = max(旧, 新)；source 不在 sources 中则追加。
            if score > target.score:
                target.score = score
            if source not in target.sources:
                target.sources.append(source)
        # claim.aspect 赋值：aspect 参数非空 → 用它（去首尾空白）；
        #   否则在 aspects 中找 name（大小写不敏感）包含于 claim 文本者取第一个；否则 ""。
        aspect_value = aspect.strip()
        if not aspect_value:
            lowered = target.text.lower()
            for a in self.aspects:
                if a.name.lower() in lowered:
                    aspect_value = a.name
                    break
        target.aspect = aspect_value
        # 任何变更后重算所有 claim 的 confidence。
        self._recompute_confidence()

    def mark_coverage(self, aspect_coverage: List[Tuple[str, bool]]) -> None:
        """用 (aspect名, 是否已覆盖) 更新覆盖矩阵。"""
        # 大小写不敏感匹配已有 aspect → covered = covered or flag（True 永不回退为 False）。
        # 未知名字 → 追加 SearchAspect(name=原名, covered=flag, required=False)。
        for raw_name, flag in aspect_coverage:
            name = raw_name.strip()
            if not name:
                continue
            key = name.lower()
            found: Optional[SearchAspect] = None
            for a in self.aspects:
                if a.name.strip().lower() == key:
                    found = a
                    break
            if found is None:
                self.aspects.append(SearchAspect(name=name, covered=bool(flag), required=False))
            else:
                found.covered = found.covered or bool(flag)

    def verify(self, claim_index: int, source: str) -> bool:
        """verify 动作：给指定主张追加独立来源。成功返回 True。"""
        # 0 <= claim_index < len(claims) 且 source 非空且不在 claim.sources →
        #   追加、重算 confidence、返回 True；否则返回 False（不做部分成功）。
        if not (0 <= claim_index < len(self.claims)):
            return False
        if not source or not source.strip():
            return False
        claim = self.claims[claim_index]
        if source in claim.sources:
            return False
        claim.sources.append(source)
        self._recompute_confidence()
        return True

    @property
    def required_uncovered(self) -> List[str]:
        """未覆盖的 required aspect 名列表（按登记顺序）。"""
        return [a.name for a in self.aspects if a.required and not a.covered]

    @property
    def coverage_ratio(self) -> float:
        """已覆盖/总数；aspects 为空 → 1.0。"""
        if not self.aspects:
            return 1.0
        covered = sum(1 for a in self.aspects if a.covered)
        return covered / len(self.aspects)

    def min_confidence(self, top_n: int = 3) -> float:
        """按 score 降序取前 top_n 条主张的最低 confidence；无主张 → 0.0。"""
        if not self.claims:
            return 0.0
        ordered = sorted(self.claims, key=lambda c: c.score, reverse=True)[:top_n]
        return min(c.confidence for c in ordered)

    def top_claims(self, n: int = 5) -> List[Tuple[int, Claim]]:
        """[(下标(0-based, 指向 self.claims), claim), ...] 按 score 降序，最多 n 条。"""
        indexed = list(enumerate(self.claims))
        indexed.sort(key=lambda pair: pair[1].score, reverse=True)
        return indexed[:n]

    def summary(self) -> str:
        """单行中文摘要，供注入 LLM 提示词。

        必须包含：覆盖 k/n、每个 aspect 的 ✓/✗ 标记、主张条数、top3 最低置信度。
        例：方面覆盖 2/3（架构✓ 性能✗ 文档✓）；主张 5 条，top3 置信度 0.71
        """
        total = len(self.aspects)
        covered = sum(1 for a in self.aspects if a.covered)
        marks = " ".join(f"{a.name}{'✓' if a.covered else '✗'}" for a in self.aspects)
        return (
            f"方面覆盖 {covered}/{total}（{marks}）；"
            f"主张 {len(self.claims)} 条，top3 置信度 {self.min_confidence(3):.2f}"
        )

    def to_trace(self) -> Dict[str, object]:
        """json.dumps 可序列化的审计快照：
        {"aspects": [{"name","covered","required"}, ...],
         "claims": [{"text","sources","score","aspect","confidence"}, ...]}"""
        return {
            "aspects": [
                {"name": a.name, "covered": a.covered, "required": a.required}
                for a in self.aspects
            ],
            "claims": [
                {
                    "text": c.text,
                    "sources": list(c.sources),
                    "score": c.score,
                    "aspect": c.aspect,
                    "confidence": c.confidence,
                }
                for c in self.claims
            ],
        }

    def _recompute_confidence(self) -> None:
        """重算所有主张置信度：
        confidence = min(1.0, 0.5 + 0.15 * (len(sources) - 1) + 0.35 * score)
        """
        for c in self.claims:
            c.confidence = min(1.0, 0.5 + 0.15 * (len(c.sources) - 1) + 0.35 * c.score)
