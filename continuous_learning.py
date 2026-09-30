"""闭环持续学习系统 (Closed-Loop Continuous Learning)

功能:
  1. Trace 记录 — 每次请求的完整上下文（四方输出、路由、检索、答案、反馈）
  2. 失败信号收集 — 用户反馈 + 自动检测
  3. 失败归因 — 规则自动归因 + LLM 辅助归因
  4. 聚类与去重 — 相似 query 聚类，保留代表案例
  5. 评估集管理 — 训练集 / 验证集 / 回归集 / 挑战集
  6. 版本化管理 — 每次优化有版本号，可回滚、可灰度
  7. 监控统计 — 失败率趋势、归因分布

关键原则:
  - 不直接改 Prompt 和模型参数，只记录
  - 优化必须有评估集，不能只看失败案例
  - 每次只改少量变量，便于归因
  - 版本化、可回滚、可灰度
"""

import json
import os
import time
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from runtime_paths import RUNTIME_ROOT
from atomic_io import atomic_write_json, atomic_append_line
from redaction import redact
from write_queue import get_default_write_queue


def _flush_writes() -> None:
    """等待已排队的状态写落盘（读路径在读之前调用，保证读到最新值）。"""
    try:
        get_default_write_queue().flush()
    except Exception:
        pass

# ---------- 目录配置 ----------
LEARNING_DIR = RUNTIME_ROOT / "learning"
TRACES_DIR = LEARNING_DIR / "traces"
FAILURE_DIR = LEARNING_DIR / "failures"
EVAL_DIR = LEARNING_DIR / "eval"
VERSIONS_DIR = LEARNING_DIR / "versions"
STATS_DIR = LEARNING_DIR / "stats"


def ensure_learning_dirs():
    for d in [TRACES_DIR, FAILURE_DIR, EVAL_DIR, VERSIONS_DIR, STATS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------- 失败类型枚举 ----------
class FailureType(str, Enum):
    INTENT_CLASSIFICATION_ERROR = "intent_classification_error"
    ROUTE_ERROR = "route_error"
    RETRIEVAL_ERROR = "retrieval_error"
    GENERATION_ERROR = "generation_error"
    STRATEGY_ERROR = "strategy_error"
    CLARIFICATION_ERROR = "clarification_error"
    OPERATION_ERROR = "operation_error"
    UNKNOWN = "unknown"


FAILURE_TYPE_LABELS = {
    FailureType.INTENT_CLASSIFICATION_ERROR: "意图分类错",
    FailureType.ROUTE_ERROR: "路由错",
    FailureType.RETRIEVAL_ERROR: "检索错",
    FailureType.GENERATION_ERROR: "生成错",
    FailureType.STRATEGY_ERROR: "策略错",
    FailureType.CLARIFICATION_ERROR: "澄清错",
    FailureType.OPERATION_ERROR: "操作错",
    FailureType.UNKNOWN: "未知",
}


# ---------- Trace 数据结构 ----------
@dataclass
class FourWayOutputs:
    llm_classifier: Dict[str, Any] = field(default_factory=dict)
    q_network: Dict[str, Any] = field(default_factory=dict)
    classification_head: Dict[str, Any] = field(default_factory=dict)
    rule_engine: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalInfo:
    queries: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)


@dataclass
class HumanLabel:
    expected_intent: Optional[str] = None
    expected_route: Optional[str] = None
    failure_type: Optional[str] = None
    notes: str = ""


@dataclass
class TraceRecord:
    trace_id: str = ""
    timestamp: float = 0.0
    user_query: str = ""
    conversation_id: str = ""
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    four_way_outputs: FourWayOutputs = field(default_factory=FourWayOutputs)
    final_intent: str = ""
    final_route: str = ""
    retrieval: RetrievalInfo = field(default_factory=RetrievalInfo)
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    token_cost: int = 0
    user_feedback: str = ""  # "positive" / "negative" / ""
    #: 产出这条 trace 的架构（"single" / "marl"）。失败样本池是两条链路共用的，
    #: 没有这个字段就无法区分「谁犯的错」，也无法按架构分别回修。
    agent: str = ""
    human_label: HumanLabel = field(default_factory=HumanLabel)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TraceRecord":
        fw = d.pop("four_way_outputs", {})
        ri = d.pop("retrieval", {})
        hl = d.pop("human_label", {})
        return cls(
            four_way_outputs=FourWayOutputs(**fw),
            retrieval=RetrievalInfo(**ri),
            human_label=HumanLabel(**hl),
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )


# ---------- 失败归因结果 ----------
@dataclass
class AttributionResult:
    trace_id: str = ""
    primary_failure: str = ""
    secondary_failures: List[str] = field(default_factory=list)
    root_cause: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0
    method: str = "rule"  # "rule" / "llm" / "human"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 评估集划分 ----------
class EvalSplit(str, Enum):
    TRAIN = "train"           # 用于生成候选 Prompt
    VALIDATION = "validation"  # 用于选择候选
    TEST = "test"             # 回归集，只用于最终评估
    CHALLENGE = "challenge"   # 边界案例、对抗案例


# ================================================================
#  核心类: ContinuousLearningEngine
# ================================================================
class ContinuousLearningEngine:
    """闭环持续学习引擎。

    纯记录 + 分析，不直接修改 Prompt 或模型参数。
    """

    def __init__(self):
        ensure_learning_dirs()
        self._trace_counter = 0
        self._load_counter()

    # ---------- 计数器 ----------
    def _counter_path(self) -> Path:
        return LEARNING_DIR / "trace_counter.json"

    def _load_counter(self):
        p = self._counter_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._trace_counter = data.get("count", 0)

    def _save_counter(self):
        atomic_write_json(self._counter_path(), {"count": self._trace_counter})

    def _next_trace_id(self) -> str:
        self._trace_counter += 1
        self._save_counter()
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"trace_{ts}_{self._trace_counter:06d}"

    # ---------- Trace 记录 ----------
    def record_trace(
        self,
        user_query: str,
        answer: str,
        conversation_id: str = "",
        intent: str = "",
        route: str = "",
        thinking: list = None,
        clarifications: list = None,
        retrieval_docs: list = None,
        retrieval_scores: list = None,
        latency_ms: float = 0.0,
        token_cost: int = 0,
        user_feedback: str = "",
        state_snapshot: dict = None,
        four_way_outputs: dict = None,
        agent: str = "",
    ) -> str:
        """记录一条完整 Trace，返回 trace_id。"""
        trace_id = self._next_trace_id()

        trace = TraceRecord(
            trace_id=trace_id,
            timestamp=time.time(),
            user_query=user_query,
            conversation_id=conversation_id,
            state_snapshot=state_snapshot or {},
            final_intent=intent,
            final_route=route,
            retrieval=RetrievalInfo(
                documents=retrieval_docs or [],
                scores=retrieval_scores or [],
            ),
            answer=answer,
            latency_ms=latency_ms,
            token_cost=token_cost,
            user_feedback=user_feedback,
            agent=agent or os.environ.get("AGENT_TYPE", ""),
        )

        # 从 thinking 提取四方输出
        if thinking:
            trace.four_way_outputs = self._extract_four_way(thinking)

        # 保存 trace
        trace_path = TRACES_DIR / f"{trace_id}.json"
        atomic_write_json(trace_path, trace.to_dict())

        # 自动检测失败信号
        failure_signals = self._detect_failure_signals(trace, thinking or [], clarifications or [])
        if failure_signals:
            self._record_failure(trace, failure_signals)

        return trace_id

    def update_feedback(self, trace_id: str, feedback: str):
        """更新用户反馈（T/F）到 trace 并触发失败记录。"""
        trace_path = TRACES_DIR / f"{trace_id}.json"
        if not trace_path.exists():
            return False

        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["user_feedback"] = feedback
        atomic_write_json(trace_path, data)

        if feedback == "negative":
            trace = TraceRecord.from_dict(data)
            signals = ["user_negative_feedback"]
            self._record_failure(trace, signals)

        return True

    def _extract_four_way(self, thinking: list) -> FourWayOutputs:
        """从 thinking steps 提取四方输出信息。"""
        fw = FourWayOutputs()
        for step in thinking:
            title = step.get("title", "")
            detail = step.get("detail", "")
            extra = step.get("extra", "")
            if "意图识别" in title:
                fw.llm_classifier = {"intent": detail, "extra": extra}
                # 尝试提取置信度
                conf_match = re.search(r"置信度\s*([\d.]+)%?", extra)
                if conf_match:
                    fw.llm_classifier["confidence"] = float(conf_match.group(1))
                # 提取路由路径
                route_match = re.search(r"路由路径\s*(\S+)", extra)
                if route_match:
                    fw.llm_classifier["route"] = route_match.group(1)
            elif "ReAct" in title:
                fw.q_network.setdefault("react_steps", []).append({
                    "title": title, "detail": detail
                })
            elif "文档检索" in title:
                fw.rule_engine = {"retrieval": detail, "extra": extra}
        return fw

    # ---------- 失败信号检测 ----------
    def _detect_failure_signals(
        self, trace: TraceRecord, thinking: list, clarifications: list
    ) -> List[str]:
        """自动检测失败信号。"""
        signals = []

        # 1. 用户反馈
        if trace.user_feedback == "negative":
            signals.append("user_negative_feedback")

        # 2. 低置信度路由
        if trace.four_way_outputs.llm_classifier:
            conf = trace.four_way_outputs.llm_classifier.get("confidence", 1.0)
            if isinstance(conf, (int, float)) and conf < 0.5:
                signals.append("low_confidence_routing")

        # 3. 澄清后仍失败 — 如果有澄清历史但最终标记为 negative
        if clarifications and trace.user_feedback == "negative":
            signals.append("clarification_still_failed")

        # 4. 检索无结果
        if trace.retrieval.documents is not None and len(trace.retrieval.documents) == 0:
            signals.append("no_retrieval_results")

        # 5. 检索低分
        if trace.retrieval.scores:
            avg_score = sum(trace.retrieval.scores) / len(trace.retrieval.scores)
            if avg_score < 0.3:
                signals.append("low_retrieval_scores")

        # 6. 答案与引用不一致（简单检查：答案中引用了文件但实际没检索到）
        if trace.retrieval.documents is not None and not trace.retrieval.documents:
            if trace.answer and ("引用" in trace.answer or "参考" in trace.answer):
                signals.append("citation_mismatch")

        # 7. 操作类失败
        for step in thinking:
            if step.get("title") == "文件操作" and "失败" in step.get("extra", ""):
                signals.append("operation_failure")
                break

        # 8. Deep Search 超轮次（ReAct 步骤过多）
        react_steps = trace.four_way_outputs.q_network.get("react_steps", [])
        if len(react_steps) > 5:
            signals.append("deep_search_over_rounds")

        # 9. 超时
        if trace.latency_ms > 30000:
            signals.append("timeout")

        return signals

    # ---------- 失败记录与归因 ----------
    def _record_failure(self, trace: TraceRecord, signals: List[str]):
        """记录失败案例并进行归因。"""
        failure_path = FAILURE_DIR / f"{trace.trace_id}.json"

        # 规则自动归因
        attribution = self._rule_attribution(trace, signals)

        failure_record = {
            "trace_id": trace.trace_id,
            "timestamp": trace.timestamp,
            # 线上 bad case 落盘即脱敏：邮箱/手机/密钥/本机用户名一律抹掉
            "user_query": redact(trace.user_query),
            "conversation_id": trace.conversation_id,
            "final_intent": trace.final_intent,
            "final_route": trace.final_route,
            "answer": redact(trace.answer, max_len=500),  # 脱敏 + 截断
            "failure_signals": signals,
            "attribution": attribution.to_dict(),
            "feedback": trace.user_feedback,
            # 归属架构：失败样本池 agent_rl / marl 共用，必须可区分
            "agent": trace.agent or os.environ.get("AGENT_TYPE", "unknown"),
        }

        atomic_write_json(failure_path, failure_record)

        # 更新统计
        self._update_stats(attribution)

    def _rule_attribution(self, trace: TraceRecord, signals: List[str]) -> AttributionResult:
        """基于规则的自动归因。"""
        result = AttributionResult(trace_id=trace.trace_id)

        if "user_negative_feedback" in signals:
            # 进一步细分
            if "low_confidence_routing" in signals:
                result.primary_failure = FailureType.INTENT_CLASSIFICATION_ERROR
                result.root_cause = "意图分类置信度低，可能分类错误"
                result.suggested_fix = "检查意图分类 Prompt 的 few-shot 示例"
            elif "no_retrieval_results" in signals or "low_retrieval_scores" in signals:
                result.primary_failure = FailureType.RETRIEVAL_ERROR
                result.root_cause = "检索未返回相关文档或相关性低"
                result.suggested_fix = "优化查询改写策略或混合检索权重"
            elif "operation_failure" in signals:
                result.primary_failure = FailureType.OPERATION_ERROR
                result.root_cause = "文件操作执行失败"
                result.suggested_fix = "检查权限、路径解析或事务逻辑"
            elif "deep_search_over_rounds" in signals:
                result.primary_failure = FailureType.STRATEGY_ERROR
                result.root_cause = "Deep Search 步骤过多，可能陷入循环"
                result.suggested_fix = "调整 ReAct 终止条件或增加步数限制"
            elif "timeout" in signals:
                result.primary_failure = FailureType.STRATEGY_ERROR
                result.root_cause = "请求处理超时"
                result.suggested_fix = "优化检索效率或增加超时后降级策略"
            elif "clarification_still_failed" in signals:
                result.primary_failure = FailureType.CLARIFICATION_ERROR
                result.root_cause = "澄清后仍无法正确处理"
                result.suggested_fix = "检查澄清 Prompt 设计和槽位填充逻辑"
            else:
                result.primary_failure = FailureType.GENERATION_ERROR
                result.root_cause = "答案生成质量不佳"
                result.suggested_fix = "检查答案生成 Prompt 和引用约束"
        elif "no_retrieval_results" in signals:
            result.primary_failure = FailureType.RETRIEVAL_ERROR
            result.root_cause = "无检索结果"
        elif "low_retrieval_scores" in signals:
            result.primary_failure = FailureType.RETRIEVAL_ERROR
            result.root_cause = "检索分数过低"
        elif "low_confidence_routing" in signals:
            result.primary_failure = FailureType.INTENT_CLASSIFICATION_ERROR
            result.root_cause = "低置信度路由"
        elif "operation_failure" in signals:
            result.primary_failure = FailureType.OPERATION_ERROR
            result.root_cause = "操作失败"
        elif "citation_mismatch" in signals:
            result.primary_failure = FailureType.GENERATION_ERROR
            result.root_cause = "答案引用不一致"
        elif "deep_search_over_rounds" in signals:
            result.primary_failure = FailureType.STRATEGY_ERROR
            result.root_cause = "超轮次"
        elif "timeout" in signals:
            result.primary_failure = FailureType.STRATEGY_ERROR
            result.root_cause = "超时"
        else:
            result.primary_failure = FailureType.UNKNOWN
            result.root_cause = "未匹配已知失败模式"

        result.secondary_failures = [s for s in signals if s != signals[0] if signals] if signals else []
        result.method = "rule"
        return result

    # ---------- 统计更新 ----------
    def _update_stats(self, attribution: AttributionResult):
        """更新失败统计。

        读-改-写是完整的竞争单元，交给单写者队列串行执行，
        否则两个线程同时读到旧 stats 后会互相覆盖（丢更新）。
        """
        try:
            get_default_write_queue().submit(lambda: self._update_stats_now(attribution))
        except RuntimeError:
            self._update_stats_now(attribution)

    def _update_stats_now(self, attribution: AttributionResult):
        """失败统计的读-改-写本体（在写线程里执行）。"""
        stats_path = STATS_DIR / "failure_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = {
                "total_traces": 0,
                "total_failures": 0,
                "by_type": {},
                "by_date": {},
                "attribution_confidence_avg": 0.0,
            }

        stats["total_failures"] += 1
        ftype = attribution.primary_failure
        stats["by_type"][ftype] = stats["by_type"].get(ftype, 0) + 1

        today = datetime.now().strftime("%Y-%m-%d")
        stats["by_date"][today] = stats["by_date"].get(today, 0) + 1

        # 更新平均置信度
        n = stats["total_failures"]
        old_avg = stats.get("attribution_confidence_avg", 0)
        stats["attribution_confidence_avg"] = old_avg + (attribution.confidence - old_avg) / n

        atomic_write_json(stats_path, stats)

    def increment_trace_count(self):
        """增加 trace 总数计数（同走单写者队列，与 _update_stats 串行）。"""
        try:
            get_default_write_queue().submit(self._increment_trace_count_now)
        except RuntimeError:
            self._increment_trace_count_now()

    def _increment_trace_count_now(self):
        stats_path = STATS_DIR / "failure_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = {"total_traces": 0, "total_failures": 0, "by_type": {}, "by_date": {}}
        stats["total_traces"] = stats.get("total_traces", 0) + 1
        atomic_write_json(stats_path, stats)

    # ---------- 聚类与去重 ----------
    def cluster_failures(self) -> Dict[str, List[dict]]:
        """对失败案例进行简单聚类（基于 query 文本相似度）。"""
        _flush_writes()
        failures = []
        for fname in sorted(FAILURE_DIR.glob("*.json")):
            with open(fname, "r", encoding="utf-8") as f:
                failures.append(json.load(f))

        if not failures:
            return {}

        # 简单聚类：基于 query 的 token 集合 Jaccard 相似度
        clusters: Dict[str, List[dict]] = {}
        assigned = set()

        for i, f1 in enumerate(failures):
            if f1["trace_id"] in assigned:
                continue
            q1_tokens = set(f1["user_query"])
            cluster_key = f1["trace_id"]
            clusters[cluster_key] = [f1]
            assigned.add(f1["trace_id"])

            for j, f2 in enumerate(failures):
                if i == j or f2["trace_id"] in assigned:
                    continue
                q2_tokens = set(f2["user_query"])
                if not q1_tokens or not q2_tokens:
                    continue
                jaccard = len(q1_tokens & q2_tokens) / len(q1_tokens | q2_tokens)
                if jaccard > 0.6:
                    clusters[cluster_key].append(f2)
                    assigned.add(f2["trace_id"])

        return clusters

    def deduplicate(self) -> int:
        """去重：相似 query 聚类后每类只保留一个代表案例。返回删除数量。"""
        clusters = self.cluster_failures()
        removed = 0
        for cluster_key, members in clusters.items():
            if len(members) <= 1:
                continue
            # 保留第一个（最早的），删除其余
            to_remove = [m["trace_id"] for m in members[1:]]
            for tid in to_remove:
                p = FAILURE_DIR / f"{tid}.json"
                if p.exists():
                    p.unlink()
                    removed += 1
        return removed

    # ---------- 评估集管理 ----------
    def add_to_eval_set(self, trace_id: str, split: EvalSplit, label: str = ""):
        """将一个 trace 加入评估集的某个分片。"""
        eval_path = EVAL_DIR / f"{split.value}.jsonl"
        record = {"trace_id": trace_id, "label": label, "added_at": datetime.now().isoformat()}
        atomic_append_line(eval_path, json.dumps(record, ensure_ascii=False))

    def get_eval_set(self, split: EvalSplit) -> List[dict]:
        """读取某个评估分片。"""
        eval_path = EVAL_DIR / f"{split.value}.jsonl"
        if not eval_path.exists():
            return []
        records = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # ---------- 版本管理 ----------
    def create_version(self, version_name: str, description: str, config: dict = None) -> str:
        """创建一个版本快照。"""
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{version_name}"
        version_dir = VERSIONS_DIR / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "version_id": version_id,
            "name": version_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "config": config or {},
            "eval_results": {},
            "status": "draft",  # draft / validated / deployed / rolled_back
        }
        atomic_write_json(version_dir / "meta.json", meta)
        # 复制当前评估集快照
        for split in EvalSplit:
            src = EVAL_DIR / f"{split.value}.jsonl"
            if src.exists():
                import shutil
                shutil.copy2(src, version_dir / f"eval_{split.value}.jsonl")

        return version_id

    def update_version_status(self, version_id: str, status: str):
        """更新版本状态。"""
        meta_path = VERSIONS_DIR / version_id / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = status
            meta["updated_at"] = datetime.now().isoformat()
            atomic_write_json(meta_path, meta)

    def list_versions(self) -> List[dict]:
        """列出所有版本。"""
        versions = []
        for d in sorted(VERSIONS_DIR.iterdir()):
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    versions.append(json.load(f))
        return versions

    # ---------- 查询接口 ----------
    def get_stats(self) -> dict:
        """获取统计数据。"""
        _flush_writes()
        stats_path = STATS_DIR / "failure_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"total_traces": 0, "total_failures": 0, "by_type": {}, "by_date": {}}

    def get_failure_cases(self, limit: int = 50) -> List[dict]:
        """获取最近的失败案例。"""
        _flush_writes()
        cases = []
        for fname in sorted(FAILURE_DIR.glob("*.json"), reverse=True)[:limit]:
            with open(fname, "r", encoding="utf-8") as f:
                cases.append(json.load(f))
        return cases

    def get_traces(self, limit: int = 100) -> List[dict]:
        """获取最近的 traces。"""
        traces = []
        for fname in sorted(TRACES_DIR.glob("*.json"), reverse=True)[:limit]:
            with open(fname, "r", encoding="utf-8") as f:
                traces.append(json.load(f))
        return traces

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """获取单条 trace。"""
        p = TRACES_DIR / f"{trace_id}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def export_failures(self, output_path: str = None) -> str:
        """导出所有失败案例到一个 JSON 文件。"""
        cases = self.get_failure_cases(limit=10000)
        if output_path is None:
            output_path = str(STATS_DIR / f"failures_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        atomic_write_json(output_path, cases)
        return output_path


# ---------- 单例 ----------
_engine_instance: Optional[ContinuousLearningEngine] = None


def get_learning_engine() -> ContinuousLearningEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ContinuousLearningEngine()
    return _engine_instance
