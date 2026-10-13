# -*- coding: utf-8 -*-
"""写意图护栏 / 澄清信号消费 / missing_slot 奖励分离 的回归测试。

对应修复：
  1. 低置信写意图不盲执行（WRITE_CONF_GUARD 护栏）           —— router
  2. 缺文件名先澄清（消费 process() 已生成的澄清信号）        —— router
  3. 澄清答案与原指令合并后重新路由                           —— router
  4. 缺槽位奖励键 missing_slot(0.0) 与 add_failure(-4.0) 分离 —— file_operators
  5. 规则路句式兜底：中英混合疑问句 → query（寒暄不进规则路，
     由检索语料 + 神经路覆盖）                                —— intent_shared/marl

既可 ``python test_write_guards.py`` 直接跑，也可被 pytest 收集。
"""

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_state import AgentState, IntentType, RoutingPath  # noqa: E402
from intent_shared import looks_like_question  # noqa: E402
from agent_rl import AgentRL, MAX_CLARIFY_DEPTH, WRITE_CONF_GUARD  # noqa: E402

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INCIDENT = "message window还是token window"

_MARL = None
_ROUTER = None


def _get_marl():
    """构造一次确定性 MARL agent（关 LLM、关采样）并复用。"""
    global _MARL
    if _MARL is None:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        import agent_rl as _agent_rl
        _agent_rl._MARL_VOTER = None
        _agent_rl._MARL_VOTER_FAILED = False
        from marl_agent import create_marl_agent
        agent = create_marl_agent()
        llm = getattr(agent, "llm_classifier", None)
        if llm is not None:
            llm.status = "unconfigured"
            if hasattr(llm, "endpoints"):
                llm.endpoints = []
        agent.deterministic = True
        _MARL = agent
    return _MARL


def _get_router():
    """构造一次 Router（single 架构，决策逻辑与架构无关）并复用。"""
    global _ROUTER
    if _ROUTER is None:
        from router import Router
        _ROUTER = Router(corpus=None, agent_type="single")
    return _ROUTER


# ---------------------------------------------------------------- 句式判据
def test_looks_like_question():
    assert looks_like_question(INCIDENT)
    assert looks_like_question("message window和token window有什么区别")
    assert looks_like_question("为什么loss不下降？")
    assert looks_like_question("what is a token")
    assert looks_like_question("is python better than java")
    assert not looks_like_question("添加文件 notes.txt 内容为 你好")
    assert not looks_like_question("把 backup.zip 删了")
    assert not looks_like_question("")
    print("[OK] looks_like_question 中英混合疑问句判据")


# ---------------------------------------------------------------- 规则路兜底
def test_rule_predict_sentence_fallback():
    agent = _get_marl()
    rule = agent.intent_agent.rule_predict

    # 疑问句式：关键词证据缺席 → query（此前弃权 → 神经路盲猜）
    assert rule(INCIDENT) == IntentType.QUERY, rule(INCIDENT)
    assert rule("为什么loss不下降") == IntentType.QUERY

    # 寒暄不进规则路（检索语料 + 神经路覆盖），规则源权重高会把
    # vague 置信度推过 p* 反触发降级，故这里必须保持弃权
    assert rule("你好呀") is None, rule("你好呀")

    # 关键词证据优先：句式兜底不得压过写操作与删除指令
    assert rule("添加文件 notes.txt 内容为 你好") == IntentType.ADD_FILE
    assert rule("删除这个文件") == IntentType.DELETE_FILE

    # 无证据且非问句仍弃权
    assert rule("xyzzy blorp") is None
    print("[OK] rule_predict 句式兜底 + 关键词证据优先")


def test_incident_classifies_not_add():
    """原始事故：LLM 缺席时中英混合问句被盲猜成 add_file。"""
    marl = _get_marl()
    state = AgentState()
    state.increment_round()
    intent, path, conf, clarifications = marl.process(INCIDENT, state)
    assert intent == IntentType.QUERY, f"expect query, got {intent} (conf={conf})"
    assert path not in (RoutingPath.FILE_ADD, RoutingPath.FILE_DELETE,
                        RoutingPath.FILE_MODIFY, RoutingPath.FILE_REFACTOR)
    print(f"[OK] 事故句不再判成写意图: {intent.value} (conf={conf:.4f})")


# ---------------------------------------------------------------- router 护栏
class _RouteStub:
    """替身记录器：替换 route 分支出口，隔离检索/写盘副作用。"""

    def __init__(self, router, process_result):
        self.calls = []
        self._r = router
        self._orig = {
            "process": router.agent_rl.process,
            "fact": router._route_fact_query,
            "op": router._route_file_operation,
            "clarify": router._route_clarification,
        }
        router.agent_rl.process = lambda q, s: process_result
        router._route_fact_query = lambda s, q: self.calls.append("fact")
        router._route_file_operation = lambda s, i, q: self.calls.append("op")
        router._route_clarification = lambda s, c: self.calls.append(("clarify", c))

    def restore(self):
        r = self._r
        r.agent_rl.process = self._orig["process"]
        r._route_fact_query = self._orig["fact"]
        r._route_file_operation = self._orig["op"]
        r._route_clarification = self._orig["clarify"]


def test_router_low_confidence_guard():
    """写意图置信度 < WRITE_CONF_GUARD → 降级事实查询，绝不盲执行。"""
    router = _get_router()
    stub = _RouteStub(router, (IntentType.ADD_FILE, RoutingPath.FILE_ADD,
                               0.15, ["您想添加哪个文件？"]))
    try:
        state = AgentState()
        state.add_dialog(INCIDENT, "")
        router.route(state)
        assert stub.calls == ["fact"], f"expect [fact], got {stub.calls}"
        assert state.current_intent == IntentType.QUERY
        assert not state.operation_history
    finally:
        stub.restore()
    print("[OK] 低置信写意图被护栏降级为事实查询")


def test_router_missing_slot_clarifies():
    """缺文件名 + 置信正常 → 消费澄清信号先问，不直接执行。"""
    router = _get_router()
    questions = ["您想添加/新建文件哪个文件？请提供文件名或路径（如 notes.txt）。"]
    stub = _RouteStub(router, (IntentType.ADD_FILE, RoutingPath.FILE_ADD,
                               0.85, questions))
    try:
        state = AgentState()
        state.add_dialog("添加文件", "")
        router.route(state)
        assert stub.calls and stub.calls[0] == ("clarify", questions), \
            f"clarification signal dropped: {stub.calls}"
    finally:
        stub.restore()
    print("[OK] 缺槽位时消费澄清信号（不再丢弃）")


def test_router_clarify_depth_exhausted_executes():
    """澄清深度封顶后不再无限问，回退到执行（缺槽位则拿 missing_slot）。"""
    router = _get_router()
    stub = _RouteStub(router, (IntentType.ADD_FILE, RoutingPath.FILE_ADD,
                               0.85, ["您想添加哪个文件？"]))
    try:
        state = AgentState()
        state.add_dialog("添加文件", "")
        state.clarification_rounds = MAX_CLARIFY_DEPTH
        router.route(state)
        assert stub.calls == ["op"], f"expect [op], got {stub.calls}"
    finally:
        stub.restore()
    print("[OK] 澄清深度封顶后回退执行")


def test_resolve_clarification_merges_answer():
    """澄清答案与原指令合并后重新路由（notes.txt 单独分类会跑偏）。"""
    router = _get_router()
    captured = {}
    orig_route = router.route
    router.route = lambda s: captured.setdefault("called", True) and s
    try:
        state = AgentState()
        state.add_dialog("添加文件", "")
        state.clarifications = ["您想添加哪个文件？"]
        router.resolve_clarification(state, "notes.txt")
        assert captured.get("called")
        assert state.dialog_history[-1].user_query == "添加文件 notes.txt", \
            state.dialog_history[-1].user_query
    finally:
        router.route = orig_route
    print("[OK] 澄清答案与原指令合并")


# ---------------------------------------------------------------- 奖励分离
def test_missing_slot_reward_key():
    router = _get_router()
    foh = router.file_operator

    s1 = AgentState()
    out = foh.add("添加文件", s1)
    assert not out.success and out.reward_key == "missing_slot", out.reward_key

    s2 = AgentState()
    out = foh.delete("删除文件", s2)
    assert not out.success and out.reward_key == "missing_slot", out.reward_key

    s3 = AgentState()
    out = foh.modify("修改文件", s3)
    assert not out.success and out.reward_key == "missing_slot", out.reward_key

    # 中性 0 与执行失败 -4 语义分离
    assert AgentRL.REWARD_MAP.get("missing_slot") == 0.0
    assert AgentRL.REWARD_MAP.get("add_failure") == -4.0
    assert router.agent_rl.compute_reward("missing_slot", None) == 0.0
    assert router.agent_rl.compute_reward("add_failure", None) == -4.0
    print("[OK] missing_slot(0.0) 与 add_failure(-4.0) 奖励分离")


def test_write_conf_guard_constant():
    assert WRITE_CONF_GUARD == 0.4
    print("[OK] WRITE_CONF_GUARD = 0.4")


ALL_TESTS = [
    test_looks_like_question,
    test_rule_predict_sentence_fallback,
    test_incident_classifies_not_add,
    test_router_low_confidence_guard,
    test_router_missing_slot_clarifies,
    test_router_clarify_depth_exhausted_executes,
    test_resolve_clarification_merges_answer,
    test_missing_slot_reward_key,
    test_write_conf_guard_constant,
]


def run_all_tests():
    failed = []
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed.append((fn.__name__, exc))
            print(f"[FAIL] {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append((fn.__name__, exc))
            print(f"[ERROR] {fn.__name__}: {type(exc).__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"  {len(failed)}/{len(ALL_TESTS)} 用例失败")
        return 1
    print(f"  全部 {len(ALL_TESTS)} 个用例通过")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
