"""失败数据回流 replay：线上 bad case 脱敏后写进经验池。

闭环::

    线上失败 (FAILURE_DIR/*.json)
        → redaction.redact() 脱敏 + 截断
        → 转成 transition（state, action, reward, next_state, done）
        → agent.replay_buffer.push()  （PER 采样，无需额外权重）
        → agent.train_step()

样本形态：
  - **负样本（必有）**：被选中的意图判错 → (s, a_wrong, 负奖励, s, done=True)
    直接把 Q(s, a_wrong) 往下拉，PER 会给它更高的采样优先级。
  - **正样本（有 ground truth 时）**：trace 的 human_label.expected_intent
    或标注字段可用 → 额外补 (s, a_correct, 正奖励, s, done=True)。

去重：按 sha1(脱敏查询 + 意图 + 来源信号) 记账到
``learning/replay_ingested.json``，同一条 bad case 只回流一次。

用法::

    from badcase_replay import push_bad_cases
    pushed, skipped = push_bad_cases(agent)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atomic_io import atomic_write_json
from redaction import redact
from runtime_paths import RUNTIME_ROOT

LEARNING_DIR = RUNTIME_ROOT / "learning"
FAILURE_DIR = LEARNING_DIR / "failures"
TRACES_DIR = LEARNING_DIR / "traces"
INGESTED_PATH = LEARNING_DIR / "replay_ingested.json"

DEFAULT_LIMIT = 500          # 单次最多回流多少条
NEGATIVE_FALLBACK = -4.0     # 没有对应失败键时的兜底负奖励

# intent → REWARD_MAP 的成功键 / 失败键
_SUCCESS_KEY = {
    "add_file": "add_success",
    "delete_file": "delete_success",
    "modify_file": "modify_success",
    "refactor": "refactor_success",
    "query": "query_success",
    "deep_search": "deep_search_success",
}
_FAILURE_KEY = {
    "add_file": "add_failure",
    "delete_file": "delete_failure",
    "modify_file": "modify_failure",
    "refactor": "refactor_failure",
}


# ================================================================ 记账
def _load_ingested() -> set:
    if not INGESTED_PATH.exists():
        return set()
    try:
        with open(INGESTED_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return set(data.get("keys", []))
    except Exception:
        return set()


def _save_ingested(keys: set) -> None:
    # 只保留最近的，避免文件无限增长
    ordered = sorted(keys)
    if len(ordered) > 20000:
        ordered = ordered[-20000:]
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(INGESTED_PATH, {"keys": ordered})


def _sample_key(query: str, intent: str, source: str) -> str:
    raw = f"{query}|{intent}|{source}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _source_name(record: Dict[str, Any]) -> str:
    """日志用的样本标识（原始文件名，避免打印已脱敏查询本身）。"""
    return str(record.get("_source") or record.get("trace_id") or "?")


# ================================================================ 读取 bad case
def load_bad_cases(limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """读取失败案例（新→旧），查询字段已脱敏。"""
    if not FAILURE_DIR.exists():
        return []
    cases: List[Dict[str, Any]] = []
    for fname in sorted(FAILURE_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(fname, "r", encoding="utf-8") as fh:
                record = json.load(fh)
        except Exception:
            continue
        record["user_query"] = redact(record.get("user_query", ""))
        if not record["user_query"]:
            continue
        record["_source"] = fname.name
        cases.append(record)
    return cases


def _expected_intent(trace_id: str, record: Dict[str, Any]) -> Optional[str]:
    """拿 ground truth：human_label > 标注字段 > None。"""
    label = record.get("expected_intent")
    if isinstance(label, str) and label:
        return label
    trace_path = TRACES_DIR / f"{trace_id}.json"
    if trace_path.exists():
        try:
            with open(trace_path, "r", encoding="utf-8") as fh:
                trace = json.load(fh)
        except Exception:
            return None
        human = trace.get("human_label") or {}
        label = human.get("expected_intent")
        if isinstance(label, str) and label:
            return label
    return None


# ================================================================ 转 transition
def _reward_for(agent, success_key: str, fallback: float) -> float:
    reward_map = getattr(agent, "REWARD_MAP", {}) or {}
    return float(reward_map.get(success_key, fallback))


def transitions_for_case(agent, record: Dict[str, Any]
                         ) -> List[Tuple[Any, int, float, Any, bool]]:
    """把一条失败记录转成 transition 列表（查询已脱敏）。"""
    import time as _time

    from agent_state import AgentState, IntentType

    query = record.get("user_query", "")
    chosen = record.get("final_intent", "")
    if not query or not chosen:
        return []

    try:
        chosen_intent = IntentType(chosen)
    except ValueError:
        return []
    if chosen_intent not in agent.reverse_map:
        return []

    state = AgentState()
    state.increment_round()
    state.time_started = _time.time()
    state.retrieval_budget = 1e9           # 与回归 gate 一致：冻结时间特征
    features = agent._extract_features(query, state)

    transitions = []
    neg = _reward_for(agent, _FAILURE_KEY.get(chosen, ""), NEGATIVE_FALLBACK)
    transitions.append(
        (features, agent.reverse_map[chosen_intent], neg, features, True)
    )

    expected = _expected_intent(record.get("trace_id", ""), record)
    if expected and expected != chosen:
        try:
            expected_intent = IntentType(expected)
        except ValueError:
            expected_intent = None
        if expected_intent is not None and expected_intent in agent.reverse_map:
            pos = _reward_for(agent, _SUCCESS_KEY.get(expected, ""), 5.0)
            transitions.append(
                (features, agent.reverse_map[expected_intent], pos, features, True)
            )

    return transitions


# ================================================================ 能力检测
def replay_capability(agent) -> Tuple[bool, str]:
    """判断该 agent 是否支持 DQN 式回流，返回 ``(支持与否, 说明)``。

    回流产物是 ``(state_vec, action, reward, next_state_vec, done)`` 五元组，
    只有 PER buffer（``replay_buffer.push``）能消费。MetaAgent 的
    ``replay_buffer`` 是裸 list、动作由 MAPPO 策略采样（还带 logprob/value），
    结构不兼容。

    旧行为：逐条尝试 → 每条都在 ``except Exception: skipped += 1; continue``
    里被静默吞掉，线上看起来「回流正常」，实际一条都没进 buffer。
    现在改为入口处一次性判定并**显式报错**。
    """
    if agent is None:
        return False, "agent 为 None"
    if not hasattr(agent, "reverse_map"):
        return False, f"{type(agent).__name__} 缺少 reverse_map"
    buf = getattr(agent, "replay_buffer", None)
    if not hasattr(buf, "push"):
        return False, (f"{type(agent).__name__}.replay_buffer 是 "
                       f"{type(buf).__name__}，没有 push()（MAPPO rollout 结构，"
                       f"与 DQN 五元组不兼容）")
    if not hasattr(agent, "train_step"):
        return False, f"{type(agent).__name__} 缺少 train_step()"
    if not hasattr(agent, "_extract_features"):
        return False, f"{type(agent).__name__} 缺少 _extract_features()"
    return True, ""


# ================================================================ 回流入口
def push_bad_cases(agent, limit: int = DEFAULT_LIMIT,
                   dry_run: bool = False,
                   train: bool = True) -> Tuple[int, int]:
    """把未回流过的 bad case 写进 replay buffer。

    Returns:
        (pushed, skipped) —— pushed 为写入的 transition 条数，
        skipped 包含「已回流过」与「无法转换」两类。
    """
    ok, reason = replay_capability(agent)
    if not ok:
        # 显式失败：调用方（main._replay_ingest / CLI）据此打印原因，
        # 不再让每个样本都各自异常一次然后被静默计数。
        print(f"  [回流] 跳过整个回流：{reason}")
        return 0, 0

    cases = load_bad_cases(limit)
    if not cases:
        return 0, 0

    ingested = _load_ingested()
    new_keys = set(ingested)
    pushed = 0
    skipped = 0

    for record in cases:
        chosen = record.get("final_intent", "")
        source_signals = ",".join(record.get("failure_signals") or []) or "unknown"
        key = _sample_key(record["user_query"], chosen, source_signals)
        if key in ingested:
            skipped += 1
            continue

        try:
            transitions = transitions_for_case(agent, record)
        except Exception as exc:
            # 转换是确定性的：同一条记录重扫多少次都会再抛一次，
            # 因此记账 + 打日志，避免每次启动都重扫同一批坏样本。
            print(f"  [回流] 记录 {_source_name(record)} 转换失败({exc})，记账跳过")
            skipped += 1
            new_keys.add(key)
            continue

        if not transitions:
            skipped += 1
            new_keys.add(key)   # 结构不合法的也记账，避免每次重扫
            continue

        if not dry_run:
            for transition in transitions:
                agent.replay_buffer.push(*transition)
            pushed += len(transitions)
        new_keys.add(key)

    if not dry_run:
        _save_ingested(new_keys)
        if pushed and train:
            try:
                agent.train_step()
            except Exception as exc:
                print(f"  [回流] 回流后触发 train_step 失败: {exc}")

    return pushed, skipped


def replay_status() -> Dict[str, Any]:
    """给 /health 之类的诊断接口用。"""
    total = len(list(FAILURE_DIR.glob("*.json"))) if FAILURE_DIR.exists() else 0
    ingested = len(_load_ingested())
    return {
        "bad_cases_total": total,
        "bad_cases_ingested": ingested,
        "bad_cases_pending": max(total - ingested, 0),
    }


if __name__ == "__main__":  # pragma: no cover - 手动排障入口
    from agent_rl import AgentRL

    agent = AgentRL(marl_vote=False)
    pushed, skipped = push_bad_cases(agent, dry_run=("--dry-run" in __import__("sys").argv))
    print(f"回流完成: pushed={pushed} skipped={skipped} buffer={len(agent.replay_buffer)}")
