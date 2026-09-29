"""config/app.yaml 的 RL 运行时参数加载（唯一读取入口）。

提供两类配置：
  1. intent_llm.reward —— 合并进 agent_rl.AgentRL.REWARD_MAP；
     键名先经 REWARD_KEY_ALIASES 归一到代码实际使用的规范键
     （add_success / query_success / deep_search_success / timeout 等）。
  2. rl: 段 —— gamma / max_rounds / retrieval_budget，
     覆盖 AgentRL.gamma、AgentState.max_rounds/retrieval_budget、
     TerminationPolicy.max_rounds/time_threshold 的内置默认值。

配置缺失或解析失败时回退到代码内置默认值，保证无配置也能启动。
"""

from typing import Any, Dict, Optional

# ---- 代码内置默认值（与 agent_rl / agent_state / evidence_evaluator 原默认一致） ----
DEFAULT_GAMMA = 0.99
DEFAULT_MAX_ROUNDS = 6
DEFAULT_RETRIEVAL_BUDGET = 300.0
DEFAULT_TIME_THRESHOLD = 120.0

# yaml 奖励键别名 → 代码实际使用的奖励键（agent_rl.REWARD_MAP / router 的 reward_key）
REWARD_KEY_ALIASES: Dict[str, str] = {
    "file_add_success": "add_success",
    "file_delete_success": "delete_success",
    "file_modify_success": "modify_success",
    "file_refactor_success": "refactor_success",
    "search_success": "query_success",
    "query_ok": "query_success",
    "deep_search_ok": "deep_search_success",
    "file_add_failure": "add_failure",
    "file_delete_failure": "delete_failure",
    "file_modify_failure": "modify_failure",
    "file_refactor_failure": "refactor_failure",
    "clarify_success": "clarification_resolved",
    "clarification_success": "clarification_resolved",
    "evidence_ok": "evidence_sufficient",
    "time_out": "timeout",
}

_CACHE: Optional[Dict[str, Any]] = None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    num = _as_float(value)
    return None if num is None else int(num)


def _normalize_rewards(raw: Dict[str, Any]) -> Dict[str, float]:
    """键名归一到代码规范键，丢弃非数值项。"""
    out: Dict[str, float] = {}
    for key, value in raw.items():
        name = REWARD_KEY_ALIASES.get(str(key).strip(), str(key).strip())
        num = _as_float(value)
        if num is None:
            continue
        out[name] = num
    return out


def load_rl_config() -> Dict[str, Any]:
    """读取 app.yaml 的 rl: 与 intent_llm.reward 段（进程内缓存，失败静默回退）。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    cfg: Dict[str, Any] = {"reward": {}}
    try:
        from providers.config_loader import get_config_loader
        app = get_config_loader().load_app_config() or {}

        rl = app.get("rl")
        if isinstance(rl, dict):
            cfg["gamma"] = _as_float(rl.get("gamma"))
            cfg["max_rounds"] = _as_int(rl.get("max_rounds"))
            cfg["retrieval_budget"] = _as_float(rl.get("retrieval_budget"))

        intent_llm = app.get("intent_llm")
        reward = intent_llm.get("reward") if isinstance(intent_llm, dict) else None
        if isinstance(reward, dict):
            cfg["reward"] = _normalize_rewards(reward)
    except Exception as e:
        print(f"  [配置] config/app.yaml 读取失败({e})，使用代码默认值。")

    _CACHE = cfg
    return cfg


def get_rl_gamma() -> float:
    value = load_rl_config().get("gamma")
    return DEFAULT_GAMMA if value is None else value


def get_max_rounds() -> int:
    value = load_rl_config().get("max_rounds")
    return DEFAULT_MAX_ROUNDS if value is None else value


def get_retrieval_budget() -> float:
    value = load_rl_config().get("retrieval_budget")
    return DEFAULT_RETRIEVAL_BUDGET if value is None else value


def get_time_threshold() -> float:
    """TerminationPolicy 的时间预算（秒）：配置存在时与 rl.retrieval_budget 同源，
    无配置时保持其历史默认值。"""
    value = load_rl_config().get("retrieval_budget")
    return DEFAULT_TIME_THRESHOLD if value is None else value


def get_reward_overrides() -> Dict[str, float]:
    """intent_llm.reward 归一后的奖励覆盖表（键已与 REWARD_MAP 对齐）。"""
    return dict(load_rl_config().get("reward") or {})
