"""回归 gate —— 固定回归集 + 基线 + 阈值 + CI 自动拦截。

定位：增量学习改造的**安全网**。任何会改动意图识别行为的变更
（奖励表、投票权重、模型 checkpoint、检索语料、Prompt），
在合入前必须先过本 gate，否则 CI 直接失败。

三个组成部分：

1. **固定回归集** ``eval/regression/regression.jsonl``
   - 确定性构造：按 ``intent_corpus.csv`` 文件顺序每类取前 N 条（不 shuffle）
   - 追加 ``AgentRL.EVAL_REAL_SAMPLES`` 的模板外真实样本
   - 内容哈希 ``sha256`` 记入基线；回归集被篡改 → gate 直接失败

2. **基线** ``eval/regression/baseline.json``（单体）/
   ``eval/regression/baseline_marl.json``（MARL）
   - 记录某次已知良好的 overall / macro / per-intent 准确率 + 回归集哈希 + 阈值
   - 两种架构各一份：准确率本就不同，共用会互相误判回退
   - 只有显式 ``--update-baseline`` 才会重写

3. **阈值判定 + CI 拦截**
   - overall 准确率较基线下降不得超过 ``tolerance``
   - macro 准确率下降不得超过 ``macro_tolerance``
   - 任一类准确率不得低于 ``min_per_intent_accuracy``
   - 任一类样本数低于 ``min_intent_samples`` → 回归集不完整，判失败
   - 非 0 退出码 → CI 拦截

用法::

    python regression_gate.py                  # 跑 gate（CI 默认行为，单体）
    python regression_gate.py --agent marl     # 跑 MARL 侧 gate
    python regression_gate.py --update-baseline # 重新固化基线（人工触发）
    python regression_gate.py --update-baseline --nsp-max-total 8 --nsp-min-infonce-acc 0.4
                                                # 固化时一并写入 NSP 阈值（探针已与训练分布对齐后再设）
    python regression_gate.py --json           # 机器可读输出
    python regression_gate.py --rebuild-set     # 重建固定回归集（会改哈希）

退出码：0 通过 / 1 未通过（回归） / 2 出错（缺基线、依赖缺失等）
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atomic_io import atomic_write_json
from runtime_paths import PROJECT_ROOT

# ---------------------------------------------------------------- 路径
# 注意：.runtime/ 与 models/ 均被 .gitignore 忽略，基线与回归集必须
# 放在受版本控制的目录下，否则 CI 每次 checkout 都拿不到基线。
GATE_DIR = PROJECT_ROOT / "eval" / "regression"
REGRESSION_SET_PATH = GATE_DIR / "regression.jsonl"
BASELINE_PATH = GATE_DIR / "baseline.json"            # 单体 AgentRL 基线
MARL_BASELINE_PATH = GATE_DIR / "baseline_marl.json"   # MARL MetaAgent 基线
CORPUS_CSV = PROJECT_ROOT / "intent_corpus.csv"

AGENT_TYPES = ("single", "marl")


def baseline_path(agent_type: str = "single") -> Path:
    """每个架构一份基线：两种策略的准确率本就不同，共用一份会互相误判回退。"""
    return MARL_BASELINE_PATH if agent_type == "marl" else BASELINE_PATH

PER_LABEL = 12          # 每类从语料库取多少条（文件顺序前 N，确定性）
TEMPLATE_LABEL_MIN = 6  # 语料库缺失的意图：用固定模板补足到该数量
MAX_TEXT_LEN = 300      # 单条样本文本长度上限

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


# ================================================================ 数据结构
@dataclass
class GateThresholds:
    """阈值配置（随基线一起持久化）。"""

    tolerance: float = 0.02           # overall 准确率相对基线最大允许跌幅
    macro_tolerance: float = 0.03     # macro 准确率相对基线最大允许跌幅
    min_per_intent_accuracy: float = 0.50   # 单类准确率下限（仅当基线自身达标时生效）
    min_per_intent_accuracy_hard: float = 0.05  # 单类绝对兜底下限（恒定生效）
    min_intent_samples: int = 5       # 单类最少样本数
    min_samples: int = 50             # 回归集最少样本数
    # ---- NSP 潜空间（None = 只记录不判定；设值才拦）----
    # probe() 是固定输入 → 固定输出，所以这些阈值跨机器可复现。
    nsp_max_total: Optional[float] = None      # NSP 总损失上限（越低越好）
    nsp_min_infonce_acc: Optional[float] = None  # InfoNCE top-1 命中率下限

    def to_dict(self) -> dict:
        return {
            "tolerance": self.tolerance,
            "macro_tolerance": self.macro_tolerance,
            "min_per_intent_accuracy": self.min_per_intent_accuracy,
            "min_per_intent_accuracy_hard": self.min_per_intent_accuracy_hard,
            "min_intent_samples": self.min_intent_samples,
            "min_samples": self.min_samples,
            "nsp_max_total": self.nsp_max_total,
            "nsp_min_infonce_acc": self.nsp_min_infonce_acc,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GateThresholds":
        base = cls()
        if not d:
            return base
        for key, value in d.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base


@dataclass
class GateMetrics:
    overall_accuracy: float = 0.0
    macro_accuracy: float = 0.0
    sample_count: int = 0
    per_intent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    set_hash: str = ""
    # NSP 潜空间的确定性探针快照（键：单体无前缀 / MARL 带角色前缀）。
    nsp: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall_accuracy": round(self.overall_accuracy, 6),
            "macro_accuracy": round(self.macro_accuracy, 6),
            "sample_count": self.sample_count,
            "per_intent": self.per_intent,
            "set_hash": self.set_hash,
            "nsp": {k: round(float(v), 6) for k, v in sorted(self.nsp.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GateMetrics":
        m = cls()
        m.overall_accuracy = float(d.get("overall_accuracy", 0.0))
        m.macro_accuracy = float(d.get("macro_accuracy", 0.0))
        m.sample_count = int(d.get("sample_count", 0))
        m.per_intent = dict(d.get("per_intent", {}))
        m.set_hash = str(d.get("set_hash", ""))
        m.nsp = {k: float(v) for k, v in dict(d.get("nsp", {})).items()}
        return m

    # ---- NSP 聚合读数（MARL 三个专家各自一份，取最差的那个）----
    def _nsp_values(self, suffix: str) -> List[float]:
        out: List[float] = []
        for key, value in self.nsp.items():
            if key == suffix or key.endswith("_" + suffix):
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    pass
        return out


@dataclass
class GateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Optional[GateMetrics] = None
    baseline: Optional[GateMetrics] = None
    thresholds: Optional[GateThresholds] = None
    error: str = ""
    agent_type: str = "single"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "agent_type": self.agent_type,
            "reasons": self.reasons,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
            "error": self.error,
        }


# ================================================================ 固定回归集
def _sha256_file(path: Path) -> str:
    """回归集内容指纹。

    按**归一化行尾**哈希：``core.autocrlf`` 会在 checkout 时把文件改成 CRLF，
    若直接对原始字节取 sha256，同一份集合在 Windows / Linux 上会得到不同指纹，
    gate 就会把「机器差异」误判成「集合被篡改」（本地 PASS、换台机器 FAIL）。
    """
    h = hashlib.sha256()
    pending = b""   # 处理跨 chunk 的 \r|\n 边界
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            data = pending + chunk
            pending = b""
            if data.endswith(b"\r"):
                data, pending = data[:-1], b"\r"
            h.update(data.replace(b"\r\n", b"\n"))
    h.update(pending)
    return h.hexdigest()


def _load_corpus_labels() -> Dict[str, List[str]]:
    """按语料库文件顺序读取 label → texts（顺序即确定性，不 shuffle）。"""
    buckets: Dict[str, List[str]] = {}
    with open(CORPUS_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            buckets.setdefault(label, []).append(text[:MAX_TEXT_LEN])
    return buckets


def build_regression_set() -> List[Dict[str, str]]:
    """确定性构造固定回归集。

    - 语料库部分：每类取文件顺序前 ``PER_LABEL`` 条（无随机）
    - 真实样本部分：``AgentRL.EVAL_REAL_SAMPLES``（模板外样本，覆盖全部意图）
    - 模板补足部分：语料库没有的意图（file_analysis / file_scan /
      report_generate 等）用 ``TASK_TEMPLATES`` 前若干条固定格式化补足，
      保证每个意图都够 ``min_intent_samples`` 条，否则类别级阈值无法判定
    - 按 text 去重，保持首次出现顺序
    """
    samples: List[Dict[str, str]] = []
    seen: set = set()

    def add(text: str, label: str) -> None:
        text = (text or "").strip()[:MAX_TEXT_LEN]
        if not text or not label or text in seen:
            return
        seen.add(text)
        samples.append({"text": text, "label": label})

    if CORPUS_CSV.exists():
        for label, texts in _load_corpus_labels().items():
            for text in texts[:PER_LABEL]:
                add(text, label)

    # 延迟导入：仅重建集合时才加载 torch
    from agent_rl import AgentRL
    for label_str, text in AgentRL.EVAL_REAL_SAMPLES.items():
        add(text, label_str.rstrip("_0123456789"))

    # 语料库缺失的意图：用固定模板补足（不 shuffle、不取随机文件名）
    counts: Dict[str, int] = {}
    for s in samples:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    files = AgentRL.SAMPLE_FILES
    content = AgentRL.SAMPLE_CONTENT
    for intent, templates in AgentRL.TASK_TEMPLATES.items():
        need = TEMPLATE_LABEL_MIN - counts.get(intent.value, 0)
        if need <= 0:
            continue
        for idx, tpl in enumerate(templates[:need]):
            try:
                text = tpl.format(f=files[idx % len(files)],
                                  g=files[(idx + 1) % len(files)],
                                  c=content)
            except (KeyError, IndexError):
                continue
            add(text, intent.value)

    return samples


def ensure_regression_set(force_rebuild: bool = False) -> Tuple[List[Dict[str, str]], str]:
    """读取（或首次创建）固定回归集，返回 (样本, sha256)。"""
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    if force_rebuild or not REGRESSION_SET_PATH.exists():
        samples = build_regression_set()
        payload = "\n".join(
            json.dumps(s, ensure_ascii=False, sort_keys=True) for s in samples
        )
        from atomic_io import atomic_write_text
        atomic_write_text(REGRESSION_SET_PATH, payload + "\n")
        samples = build_regression_set()  # 重新读盘，保证哈希与磁盘一致

    with open(REGRESSION_SET_PATH, "r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]
    return samples, _sha256_file(REGRESSION_SET_PATH)


# ================================================================ 确定性预测
class _NoLLM:
    """禁用 LLM 路：CI 无 key，且 LLM 采样不可复现。"""

    def classify(self, query: str):
        return None


def seed_everything(seed: int = 0) -> None:
    """固定所有随机源。

    必须在 **构造 AgentRL 之前** 调用：没有 checkpoint 时网络是随机初始化的，
    不固定种子则每次进程的权重都不同，gate 无法复现。
    """
    import random as _random
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():  # pragma: no cover - CI 无 GPU
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # 让子进程也继承固定哈希种子（set/dict 迭代顺序才稳定）
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextlib.contextmanager
def deterministic_mode(agent):
    """把 agent 切到可复现的推理模式：ε=0、关 LLM、关交叉投票、关 dropout。

    单体（AgentRL）与 MARL（MetaAgent）属性集不同，这里按能力探测，
    两个架构都能进入确定性模式 —— 以前只按 AgentRL 写死，MARL 根本进不了 gate。
    """
    saved_epsilon = getattr(agent, "epsilon", None)
    saved_marl_vote = getattr(agent, "marl_vote", None)
    saved_llm = getattr(agent, "llm_classifier", None)
    saved_deterministic = getattr(agent, "deterministic", None)

    nets = []
    for name in ("q_network", "classifier"):
        net = getattr(agent, name, None)
        if net is not None:
            nets.append(net)
    for role in ("intent_agent", "subintent_agent", "slot_agent"):
        pa = getattr(agent, role, None)
        if pa is not None:
            for name in ("q_network", "critic"):
                net = getattr(pa, name, None)
                if net is not None:
                    nets.append(net)
    engram = getattr(agent, "engram", None)
    if engram is not None:
        nets.append(engram)
    nets = [n for n in nets if hasattr(n, "eval")]

    for net in nets:
        net.eval()
    if saved_epsilon is not None:
        agent.epsilon = 0.0
    if saved_marl_vote is not None:
        agent.marl_vote = False
    if saved_deterministic is not None:
        # MetaAgent: 关闭策略采样，否则同一样本两次跑出不同意图
        agent.deterministic = True
    if saved_llm is not None:
        agent.llm_classifier = _NoLLM()
    seed_everything(0)
    try:
        yield
    finally:
        if saved_epsilon is not None:
            agent.epsilon = saved_epsilon
        if saved_marl_vote is not None:
            agent.marl_vote = saved_marl_vote
        if saved_deterministic is not None:
            agent.deterministic = saved_deterministic
        if saved_llm is not None:
            agent.llm_classifier = saved_llm
        for net in nets:
            if hasattr(net, "train"):
                net.train()


def evaluate_set(agent, samples: List[Dict[str, str]]) -> GateMetrics:
    """在固定回归集上跑一遍意图识别，产出准确率指标。"""
    from agent_state import AgentState, IntentType

    valid_labels = {it.value for it in IntentType}
    per_intent: Dict[str, Dict[str, Any]] = {}
    correct_total = 0
    used = 0

    with deterministic_mode(agent), _torch_no_grad():
        for sample in samples:
            text, label = sample.get("text", ""), sample.get("label", "")
            if label not in valid_labels:
                continue
            state = AgentState()
            state.increment_round()
            # 冻结时间维度：state_vector 含 elapsed_time/retrieval_budget，
            # 若随墙钟漂移，同一条样本在不同次运行会得到不同特征。
            # 把预算拉到极大 → 该项恒等于 0（低于 float32 精度）。
            state.time_started = time.time()
            state.retrieval_budget = 1e9
            try:
                predicted, _conf = agent.classify_intent(text, state)
            except Exception:
                predicted = None
            used += 1
            bucket = per_intent.setdefault(
                label, {"correct": 0, "total": 0, "accuracy": 0.0,
                        "misses": []}
            )
            bucket["total"] += 1
            ok = predicted is not None and predicted.value == label
            if ok:
                bucket["correct"] += 1
                correct_total += 1
            elif len(bucket["misses"]) < 5:
                bucket["misses"].append({
                    "text": text[:60],
                    "predicted": predicted.value if predicted else None,
                })

    for bucket in per_intent.values():
        bucket["accuracy"] = (
            round(bucket["correct"] / bucket["total"], 6) if bucket["total"] else 0.0
        )

    accs = [b["accuracy"] for b in per_intent.values() if b["total"]]
    return GateMetrics(
        overall_accuracy=(correct_total / used) if used else 0.0,
        macro_accuracy=(sum(accs) / len(accs)) if accs else 0.0,
        sample_count=used,
        per_intent=per_intent,
    )


# NSP 探针里这几个键是「模型学没学到东西」的直接读数，报告只列它们
NSP_CORE_KEYS = ("nsp_total", "nsp_infonce_acc", "nsp_rollout_err", "ema_drift")


def probe_nsp(agent) -> Dict[str, float]:
    """跑一次 NSP 确定性探针。

    探针内部固定 RNG + eval，同一份权重两次调用逐位一致，所以它的读数
    能和准确率一样写进基线。agent 没接 NSP（旧对象 / 构造失败）→ 空字典，
    不能因为多了一项指标就把整个 gate 打成基础设施错误。
    """
    probe = getattr(agent, "nsp_probe", None)
    if probe is None:
        return {}
    try:
        return {k: round(float(v), 6) for k, v in probe().items()}
    except Exception:
        return {}


class _torch_no_grad:
    """无 torch 环境下也能跑；有 torch 时进入 no_grad。"""

    def __enter__(self):
        try:
            import torch
            self._ctx = torch.no_grad()
            return self._ctx.__enter__()
        except Exception:
            self._ctx = None
            return None

    def __exit__(self, *exc):
        if self._ctx is not None:
            return self._ctx.__exit__(*exc)
        return False


# ================================================================ 基线
def _reset_retriever() -> None:
    """丢弃已构造的检索索引，让它在下次查询时从语料重建。

    gate 的结论必须可跨机器复现：本地可能命中几周前的缓存（甚至是旧版
    分词/编号逻辑留下的），CI 则永远是冷启动。强制重建把两边拉到同一条起跑线。
    """
    try:
        import intent_retriever as _ir
        if getattr(_ir, "_RETRIEVER", None) is not None:
            _ir._RETRIEVER.rebuild(force=True)
    except Exception as exc:  # 重建失败不掩盖 gate 自身的判定
        print(f"  [gate] 检索索引重建失败({exc})，继续用现有索引。")


def load_baseline(agent_type: str = "single"
                  ) -> Tuple[Optional[GateMetrics], Optional[GateThresholds]]:
    path = baseline_path(agent_type)
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return (
        GateMetrics.from_dict(data.get("metrics", {})),
        GateThresholds.from_dict(data.get("thresholds")),
    )


def save_baseline(metrics: GateMetrics,
                  thresholds: Optional[GateThresholds] = None,
                  agent_type: str = "single") -> Path:
    thresholds = thresholds or GateThresholds()
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "agent_type": agent_type,
        "regression_set": str(REGRESSION_SET_PATH.relative_to(PROJECT_ROOT)),
        "metrics": metrics.to_dict(),
        "thresholds": thresholds.to_dict(),
        "note": "由 regression_gate.py --update-baseline 生成，请勿手工编辑",
    }
    path = baseline_path(agent_type)
    atomic_write_json(path, payload)
    return path


# ================================================================ gate 判定
def check(metrics: GateMetrics, baseline: GateMetrics,
          thresholds: GateThresholds, set_hash: str,
          agent_type: str = "single",
          nsp_enforce: bool = True) -> GateResult:
    reasons: List[str] = []

    if metrics.sample_count < thresholds.min_samples:
        reasons.append(
            f"回归集样本不足：{metrics.sample_count} < {thresholds.min_samples}"
        )

    # ---- NSP 潜空间：阈值配了才拦；配了却读不到数 → 指标本身失效，同样拦 ----
    # nsp_enforce=False（agent 是新鲜随机初始化、没加载到 checkpoint）时只
    # 记录不判定：随机权重的探针读数反映的是初始化分布而非训练质量，此时硬拦
    # 只会让 CI（models/ 被 gitignore，checkout 后必然无权重）恒失败。
    # 本地有训练权重时 nsp_enforce=True，阈值正常生效。
    if nsp_enforce and thresholds.nsp_max_total is not None:
        totals = metrics._nsp_values("nsp_total")
        if not totals:
            reasons.append("NSP 阈值已配置但探针没有 nsp_total 读数（指标失效）")
        elif max(totals) > thresholds.nsp_max_total:
            reasons.append(
                f"NSP 总损失 {max(totals):.4f} 超过上限 "
                f"{thresholds.nsp_max_total:.4f}"
            )
    if nsp_enforce and thresholds.nsp_min_infonce_acc is not None:
        accs = metrics._nsp_values("nsp_infonce_acc")
        if not accs:
            reasons.append("NSP 阈值已配置但探针没有 nsp_infonce_acc 读数（指标失效）")
        elif min(accs) < thresholds.nsp_min_infonce_acc:
            reasons.append(
                f"NSP InfoNCE 命中率 {min(accs):.4f} 低于下限 "
                f"{thresholds.nsp_min_infonce_acc:.4f}"
            )

    if baseline.set_hash and set_hash != baseline.set_hash:
        reasons.append(
            "固定回归集内容已被修改（sha256 不匹配基线）；"
            "如确为有意扩充，请执行 python regression_gate.py --update-baseline"
        )

    delta = metrics.overall_accuracy - baseline.overall_accuracy
    if delta < -thresholds.tolerance:
        reasons.append(
            f"overall 准确率回退 {abs(delta):.4f} > 容忍 {thresholds.tolerance:.4f} "
            f"(baseline={baseline.overall_accuracy:.4f}, now={metrics.overall_accuracy:.4f})"
        )

    macro_delta = metrics.macro_accuracy - baseline.macro_accuracy
    if macro_delta < -thresholds.macro_tolerance:
        reasons.append(
            f"macro 准确率回退 {abs(macro_delta):.4f} > 容忍 {thresholds.macro_tolerance:.4f} "
            f"(baseline={baseline.macro_accuracy:.4f}, now={metrics.macro_accuracy:.4f})"
        )

    for intent, bucket in sorted(metrics.per_intent.items()):
        base_bucket = baseline.per_intent.get(intent)
        acc = float(bucket.get("accuracy", 0.0))
        total = int(bucket.get("total", 0))

        if total < thresholds.min_intent_samples:
            reasons.append(
                f"类别 {intent} 样本不足：{total} < {thresholds.min_intent_samples}"
            )
            continue

        # 绝对兜底下限：恒定生效，拦住「类几乎全错」
        if acc < thresholds.min_per_intent_accuracy_hard:
            reasons.append(
                f"类别 {intent} 准确率 {acc:.4f} 低于绝对下限 "
                f"{thresholds.min_per_intent_accuracy_hard:.4f}"
            )
            continue

        if base_bucket is None:
            # 新增类别（基线里没有）：只要求达到绝对下限
            continue

        base_acc = float(base_bucket.get("accuracy", 0.0))
        if acc < base_acc - thresholds.tolerance:
            reasons.append(
                f"类别 {intent} 准确率回退 {base_acc - acc:.4f} > 容忍 "
                f"{thresholds.tolerance:.4f} (baseline={base_acc:.4f}, now={acc:.4f})"
            )
            continue

        # 相对下限：仅当基线自身达标时才要求本次同样达标。
        # 基线本身就低于该下限的类属于「已知欠账」，只要求不再回退。
        if (base_acc >= thresholds.min_per_intent_accuracy
                and acc < thresholds.min_per_intent_accuracy):
            reasons.append(
                f"类别 {intent} 准确率 {acc:.4f} 低于下限 "
                f"{thresholds.min_per_intent_accuracy:.4f}"
            )

    return GateResult(
        passed=not reasons,
        reasons=reasons,
        metrics=metrics,
        baseline=baseline,
        thresholds=thresholds,
        agent_type=agent_type,
    )


def _build_agent(agent_type: str):
    """按架构构造 gate 用的 agent（与线上默认配置一致）。"""
    seed_everything(0)
    if agent_type == "marl":
        from marl_agent import create_marl_agent
        return create_marl_agent()
    from agent_rl import AgentRL
    return AgentRL(marl_vote=False)


def run_gate(update_baseline: bool = False,
             rebuild_set: bool = False,
             agent=None,
             agent_type: str = "single",
             nsp_max_total: Optional[float] = None,
             nsp_min_infonce_acc: Optional[float] = None) -> GateResult:
    """执行 gate。``update_baseline=True`` 时跑完直接固化为新基线并返回通过。

    ``agent_type`` 选 ``single``（AgentRL）或 ``marl``（MetaAgent）。以前只测单体，
    MARL 侧的回归根本不会被门禁发现。

    ``nsp_max_total`` / ``nsp_min_infonce_acc`` 仅在 ``update_baseline=True``
    时生效：把本次阈值一起写入基线（只改阈值不改指标）。不传则沿用基线里的
    旧阈值（首次固化时为 None = 只记录不判定）。
    """
    if agent_type not in AGENT_TYPES:
        return GateResult(passed=False, agent_type=agent_type,
                          error=f"未知 agent_type: {agent_type}（可选 {AGENT_TYPES}）")

    try:
        samples, set_hash = ensure_regression_set(force_rebuild=rebuild_set)
    except Exception as exc:
        return GateResult(passed=False, agent_type=agent_type,
                          error=f"构建固定回归集失败: {exc}")

    if not samples:
        return GateResult(passed=False, agent_type=agent_type,
                          error="固定回归集为空")

    if agent is None:
        try:
            # 先固定种子再建 agent：无 checkpoint 时网络是随机初始化的
            agent = _build_agent(agent_type)
        except Exception as exc:
            return GateResult(passed=False, agent_type=agent_type,
                              error=f"加载 {agent_type} agent 失败: {exc}")

    # gate 必须与本地缓存状态无关：检索索引强制从语料现场重建，
    # 否则「本地命中旧缓存」和「CI 冷启动重建」会给出不同的逐条预测。
    _reset_retriever()

    try:
        metrics = evaluate_set(agent, samples)
    except Exception as exc:
        return GateResult(passed=False, agent_type=agent_type,
                          error=f"评估执行失败: {exc}")
    metrics.set_hash = set_hash
    metrics.nsp = probe_nsp(agent)

    baseline, thresholds = load_baseline(agent_type)
    thresholds = thresholds or GateThresholds()

    if update_baseline:
        if nsp_max_total is not None:
            thresholds.nsp_max_total = float(nsp_max_total)
        if nsp_min_infonce_acc is not None:
            thresholds.nsp_min_infonce_acc = float(nsp_min_infonce_acc)
        save_baseline(metrics, thresholds, agent_type)
        return GateResult(passed=True, metrics=metrics,
                          baseline=metrics, thresholds=thresholds,
                          agent_type=agent_type)

    if baseline is None:
        rel = baseline_path(agent_type)
        return GateResult(
            passed=False, metrics=metrics, thresholds=thresholds,
            agent_type=agent_type,
            error=f"缺少基线文件（{rel}）；首次运行请执行 "
                  f"python regression_gate.py --agent {agent_type} --update-baseline",
        )

    # 新鲜随机初始化（没加载到 checkpoint）时 NSP 只记录不判定，见 check()。
    # models/ 被 gitignore，CI checkout 后必然无权重；不断言这一点就会恒失败。
    nsp_enforce = bool(getattr(agent, "loaded_from_disk", False))
    return check(metrics, baseline, thresholds, set_hash, agent_type,
                 nsp_enforce=nsp_enforce)


# ================================================================ CLI
def _format_report(result: GateResult) -> str:
    lines: List[str] = []
    lines.append("=" * 64)
    lines.append(f"回归 gate (regression gate) — agent={result.agent_type}")
    lines.append("=" * 64)
    if result.error:
        lines.append(f"[ERROR] {result.error}")
    if result.metrics:
        m = result.metrics
        lines.append(f"当前  overall={m.overall_accuracy:.4f}  "
                     f"macro={m.macro_accuracy:.4f}  n={m.sample_count}")
        floor = (result.thresholds.min_per_intent_accuracy_hard
                 if result.thresholds else 0.0)
        for intent, b in sorted(m.per_intent.items()):
            mark = "OK " if b["accuracy"] >= floor else "LOW"
            lines.append(f"   [{mark}] {intent:<16} {b['accuracy']:.4f}  "
                         f"({b['correct']}/{b['total']})")
    if result.baseline:
        b = result.baseline
        lines.append(f"基线  overall={b.overall_accuracy:.4f}  "
                     f"macro={b.macro_accuracy:.4f}  n={b.sample_count}")
    if result.metrics and result.metrics.nsp:
        lines.append("NSP   确定性探针（同权重复现；阈值未配置时只记录不判定）")
        for key, value in sorted(result.metrics.nsp.items()):
            if key.endswith(NSP_CORE_KEYS):
                lines.append(f"   {key:<30} {value:.6f}")
        base_nsp = result.baseline.nsp if result.baseline else {}
        if base_nsp:
            lines.append("基线 NSP")
            for key, value in sorted(base_nsp.items()):
                if key.endswith(NSP_CORE_KEYS):
                    lines.append(f"   {key:<30} {value:.6f}")
    if result.reasons:
        lines.append("-" * 64)
        for reason in result.reasons:
            lines.append(f"[FAIL] {reason}")
    lines.append("-" * 64)
    lines.append("PASS" if result.passed else "FAIL")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="回归 gate：固定回归集 + 基线 + 阈值，未通过则非 0 退出"
    )
    parser.add_argument("--update-baseline", action="store_true",
                        help="跑一遍并把当前指标固化为新基线")
    parser.add_argument("--rebuild-set", action="store_true",
                        help="重建固定回归集（会改变内容哈希）")
    parser.add_argument("--agent", choices=list(AGENT_TYPES), default="single",
                        help="被测架构：single=AgentRL（默认），marl=MetaAgent；"
                             "两者各有一份独立基线")
    parser.add_argument("--nsp-max-total", type=float, default=None,
                        help="与 --update-baseline 联用：把 NSP 总损失上限写入基线阈值")
    parser.add_argument("--nsp-min-infonce-acc", type=float, default=None,
                        help="与 --update-baseline 联用：把 InfoNCE 命中率下限写入基线阈值")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="输出 JSON 报告")
    parser.add_argument("--quiet", action="store_true", help="只输出结论")
    args = parser.parse_args(argv)

    # 只改阈值不改指标：不带 --update-baseline 时，两个 NSP 阈值参数只重写
    # 基线文件里的 thresholds 段，metrics（准确率基线）原样保留。
    # 用于“准确率基线不动、只让 NSP 从只记录变为硬判定”的场景。
    if not args.update_baseline and (
        args.nsp_max_total is not None or args.nsp_min_infonce_acc is not None
    ):
        baseline, thresholds = load_baseline(args.agent)
        if baseline is None:
            print(f"缺少基线文件，先跑 python regression_gate.py "
                  f"--agent {args.agent} --update-baseline")
            return EXIT_ERROR
        thresholds = thresholds or GateThresholds()
        if args.nsp_max_total is not None:
            thresholds.nsp_max_total = float(args.nsp_max_total)
        if args.nsp_min_infonce_acc is not None:
            thresholds.nsp_min_infonce_acc = float(args.nsp_min_infonce_acc)
        save_baseline(baseline, thresholds, args.agent)
        print(f"已更新 {baseline_path(args.agent).name} 阈值："
              f"nsp_max_total={thresholds.nsp_max_total}, "
              f"nsp_min_infonce_acc={thresholds.nsp_min_infonce_acc}（指标未动）")
        return EXIT_PASS

    try:
        result = run_gate(update_baseline=args.update_baseline,
                          rebuild_set=args.rebuild_set,
                          agent_type=args.agent,
                          nsp_max_total=args.nsp_max_total,
                          nsp_min_infonce_acc=args.nsp_min_infonce_acc)
    except KeyboardInterrupt:
        return EXIT_ERROR

    if args.as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif args.quiet:
        print("ERROR" if result.error else ("PASS" if result.passed else "FAIL"))
    else:
        print(_format_report(result))

    # 退出码契约：0=通过 / 1=回归（check 判出的 reasons）/ 2=基础设施错误
    # （缺基线、回归集构建失败、agent 加载失败）。以前只有「没有 metrics」
    # 才返回 2，缺基线时 metrics 已经算出来了 → 落到 1，与文档的
    # 「2=缺基线」不一致，调用方无法区分「回归了」和「跑不起来」。
    if result.error:
        return EXIT_ERROR
    return EXIT_PASS if result.passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
