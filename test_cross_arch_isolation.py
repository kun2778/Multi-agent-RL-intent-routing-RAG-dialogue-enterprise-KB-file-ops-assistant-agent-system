"""跨架构结构性放大器的回归护栏。

评审点出的两类放大器（原文 section 3）在这里被钉死：

1. **循环交叉投票** —— ``agent_rl.get_marl_voter`` ↔ ``MetaAgent.single_cross_vote``
   互相嵌套。只要其中一侧的开关被改破，就会出现
   marl → agent_rl → marl → agent_rl 的构造回环（旧实现用非重入
   ``threading.Lock``，撞上回环会直接死锁而不是报错）。
2. **共享上游/下游灌入** —— 两架构共用 ``get_intent_retriever()`` 单例、
   同一套 QueryRewriter / LLMIntent 类。跨架构的那一票**只允许读本架构的
   模型参数**，一旦它重新跑共享上游，同一个上游错误就会被计两次，
   一侧的误判变成另一侧的「独立佐证」。
"""

import contextlib
import functools
import json
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent_state import AgentState
import agent_rl
from marl_agent import create_marl_agent, create_marl_voter


def _boom(*args, **kwargs):
    raise AssertionError("跨架构票重新进入了共享上游（LLM/改写器/检索器）")


@contextlib.contextmanager
def _fresh_voter_slot():
    """临时清空第 6 路惰性单例，退出时原样还原（避免污染其它用例）。"""
    saved_voter = agent_rl._MARL_VOTER
    saved_failed = agent_rl._MARL_VOTER_FAILED
    agent_rl._MARL_VOTER = None
    agent_rl._MARL_VOTER_FAILED = False
    try:
        yield
    finally:
        agent_rl._MARL_VOTER = saved_voter
        agent_rl._MARL_VOTER_FAILED = saved_failed
        agent_rl._MARL_VOTER_BUILDING = False


#: 跨架构票必须完全绕开的共享上游入口
_SHARED_UPSTREAM = {
    "llm_classifier": ("classify",),
    "intent_retriever": ("vote",),
    "query_rewriter": ("rewrite", "classify_query_subintent", "top_operation",
                       "extract_slots", "generate_clarification", "extract_target"),
}


@contextlib.contextmanager
def _upstream_is_poisoned(*agents):
    """把共享上游入口全部替换成「一调用就炸」的桩（退出时自动还原）。"""
    stack = contextlib.ExitStack()
    with stack:
        for agent in agents:
            for comp_name, methods in _SHARED_UPSTREAM.items():
                comp = getattr(agent, comp_name, None)
                if comp is None:
                    continue
                for meth in methods:
                    if hasattr(comp, meth):
                        stack.enter_context(
                            mock.patch.object(comp, meth, side_effect=_boom))
        yield


#: 上游相关用例共用一个 MARL 实例（构造 MetaAgent + AgentRL 很贵，
#: 每个用例各造一次会把整套测试拖慢十几秒）
@functools.lru_cache(maxsize=1)
def _marl():
    return create_marl_agent()


# ================================================================
# 1. 循环交叉投票
# ================================================================
def test_marl_voter_construction_is_acyclic_singleton():
    """第 6 路构造不变式：无环 + 惰性单例 + 重入拒绝。

    - 两个工厂造出来的对象构成 DAG 而非环；
    - ``get_marl_voter`` 每进程只构造一次并复用；
    - 构造过程中的重入直接返回 None（旧实现用非重入锁会死锁）。
    """
    agent = create_marl_agent()
    assert agent.single_cross_vote is True, "完整 MARL 必须开交叉路"
    inner = agent.single_voter
    assert inner is not None, "交叉路的 AgentRL 应当可用"
    assert inner.marl_vote is False, \
        "交叉路的 AgentRL 不得再开第 6 路（否则 agent_rl ↔ marl 嵌套）"

    voter = create_marl_voter()
    assert voter.single_cross_vote is False, \
        "第 6 路投票器必须关掉交叉路，否则它内部会再造一个 AgentRL"
    assert voter.single_voter is None, \
        "第 6 路投票器不得内嵌单 Agent 交叉路"

    with _fresh_voter_slot():
        first = agent_rl.get_marl_voter()
        assert first is not None, "第 6 路应当能构造出来"
        assert agent_rl.get_marl_voter() is first, "第 6 路必须复用同一实例"
        agent_rl._MARL_VOTER_BUILDING = True
        assert agent_rl.get_marl_voter() is None, \
            "构造回环时应放弃第 6 路，而不是继续构造"
    print("[OK] 跨架构构造无环、第 6 路为惰性单例、重入被拒绝")


# ================================================================
# 2. 共享上游灌入
# ================================================================
def test_shared_upstream_topology_is_pinned():
    """钉死共享拓扑：检索器是刻意共享的单例，改写器/LLM 不共享实例。

    一旦有人把 ``QueryRewriter`` 也改成单例，一边的改写缓存就会直接
    灌进另一边的投票，本用例会先失败。
    """
    agent = _marl()
    inner = agent.single_voter

    assert agent.intent_retriever is inner.intent_retriever, \
        "检索器本就是共享单例（只读检索，允许共享）"
    assert agent.query_rewriter is not inner.query_rewriter, \
        "改写器不得共享实例：一边的改写/插槽缓存会污染另一边"
    assert agent.llm_classifier is not inner.llm_classifier, \
        "LLM 客户端不得共享实例：端点/限流状态会互相灌入"
    print("[OK] 共享上游拓扑符合设计（检索共享、改写/LLM 各自独立）")


def test_cross_votes_never_re_enter_shared_upstream():
    """两架构的跨架构票只读本架构模型参数，绝不重跑共享上游。

    这是「共享上游灌入」的核心防线：若 ``MetaAgent.vote``（给单体的第 6 票）
    或 ``AgentRL.neural_vote``（给 MARL 的 single 票）重新调用 LLM、
    改写器或检索器，同一个上游错误就会被两侧各计一次，
    看起来像两票独立佐证。
    """
    agent = _marl()
    inner = agent.single_voter
    state = AgentState()
    state.increment_round()
    query = "对比分析 a.txt 和 b.txt"

    with _upstream_is_poisoned(agent, inner):
        marl_intent, marl_conf, marl_info = agent.vote(query, state)
        single_intent, single_conf, _ = inner.neural_vote(query, state)

    assert marl_intent is not None and 0.0 <= marl_conf <= 1.0
    assert single_intent is not None and 0.0 <= single_conf <= 1.0
    print(f"[OK] 跨架构票绕开共享上游: marl={marl_intent.value}/{marl_conf:.3f}, "
          f"single={single_intent.value}/{single_conf:.3f}")


# ================================================================
# 3. 存量失败记录回填
# ================================================================
def test_backfill_failure_agent_fills_missing_only(tmp_path):
    """存量 ``learning/failures/*.json`` 缺 agent 字段必须被补齐且幂等。"""
    import continuous_learning as cl

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"trace_id": "a", "user_query": "x"}),
                      encoding="utf-8")
    fresh = tmp_path / "fresh.json"
    fresh.write_text(json.dumps({"trace_id": "b", "agent": "marl"}),
                     encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    patched = cl.backfill_failure_agent(directory=tmp_path, default_agent="single")
    assert patched == 1, f"只应补 1 条缺字段的记录，实为 {patched}"
    assert json.loads(legacy.read_text(encoding="utf-8"))["agent"] == "single"
    assert json.loads(fresh.read_text(encoding="utf-8"))["agent"] == "marl"

    again = cl.backfill_failure_agent(directory=tmp_path, default_agent="single")
    assert again == 0, "回填必须幂等，不得重复改写"
    print("[OK] 存量失败记录补 agent 字段（只补缺失、幂等）")


def test_backfill_defaults_to_unknown_when_nothing_is_known(tmp_path):
    """既无传入值也无环境变量时，用显式 unknown 而不是留缺失键。"""
    import continuous_learning as cl

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"trace_id": "a"}), encoding="utf-8")
    saved_env = os.environ.pop("AGENT_TYPE", None)
    try:
        patched = cl.backfill_failure_agent(directory=tmp_path, default_agent=None)
    finally:
        if saved_env is not None:
            os.environ["AGENT_TYPE"] = saved_env
    assert patched == 1
    assert json.loads(legacy.read_text(encoding="utf-8"))["agent"] == "unknown"
    print("[OK] 无法归因的旧记录落为 unknown（显式，而非缺失）")


def run_all_tests():
    print("\n" + "#" * 70)
    print("  跨架构结构性放大器护栏")
    print("#" * 70)
    test_marl_voter_construction_is_acyclic_singleton()
    test_shared_upstream_topology_is_pinned()
    test_cross_votes_never_re_enter_shared_upstream()
    return True


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        test_backfill_failure_agent_fills_missing_only(Path(td))
        test_backfill_defaults_to_unknown_when_nothing_is_known(Path(td))
    run_all_tests()
