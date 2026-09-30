"""Test script for MARL (Multi-Agent Reinforcement Learning) system.

同时跑两套架构（MARL / 单体）并**断言**两者都预测正确。

历史问题：本文件原先只 ``print``「OK / DIFF」、只按 MARL 的结果判定通过，
单体的预测只打印不参与判定 —— 于是「两个 agent 一起犯错」永远不会让
测试失败。现在两架构各自独立判定，任一边出错都让用例失败。
"""

import sys
import os
import random

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_state import AgentState
from marl_agent import create_marl_agent
from agent_rl import AgentRL

# Fix encoding for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')


def _seed_all(seed: int = 0) -> None:
    """固定随机源：models/ 下无 checkpoint 时网络是随机初始化的，
    不固定种子则每次进程权重不同，断言会变成掷骰子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# pytest 先导入全部测试模块、再执行用例：在导入期播种能覆盖后续所有构造。
_seed_all(0)


# Test cases: (query, expected_intent)
TEST_CASES = [
    # Original problematic query
    ("帮我把最近的截屏内容的文字进行识别并输出", "query"),

    # OCR-related queries
    ("识别截屏中的文字", "query"),
    ("提取图片内容", "query"),

    # File operations
    ("添加文件 notes.txt 内容为 你好", "add_file"),
    ("删除文件 old.txt", "delete_file"),
    ("修改文件 config.yaml", "modify_file"),
    ("整理文件夹", "refactor"),

    # Query operations
    ("搜索强化学习", "query"),
    ("查看这个文件", "query"),

    # Deep search
    ("对比分析 a.txt 和 b.txt", "deep_search"),

    # Vague
    ("你好", "vague"),
]

_AGENTS = {}


def _disable_llm(agent):
    """测试不依赖网络：LLM 路固定为未配置，结果可复现。"""
    llm = getattr(agent, "llm_classifier", None)
    if llm is not None:
        llm.status = "unconfigured"
        if hasattr(llm, "endpoints"):
            llm.endpoints = []
    return agent


def get_agents():
    """两套架构各构造一次并复用（重复构造会反复加载模型）。"""
    if not _AGENTS:
        _seed_all(0)
        # 第 6 路投票器是 agent_rl 模块级单例，可能已被其它用例在随机状态下
        # 构造过；重置后按固定种子重建，否则本测试的结果不可复现。
        import agent_rl as _agent_rl
        _agent_rl._MARL_VOTER = None
        _agent_rl._MARL_VOTER_FAILED = False

        marl_agent = _disable_llm(create_marl_agent())
        marl_agent.deterministic = True          # 关掉策略采样

        single_agent = _disable_llm(AgentRL())
        single_agent.epsilon = 0.0               # 关掉 ε 探索

        _AGENTS["marl"] = marl_agent
        _AGENTS["single"] = single_agent
    return _AGENTS["marl"], _AGENTS["single"]


def predict_both(query: str):
    """对同一条 query 同时跑两套架构，返回 (marl 意图, 单体 意图)。"""
    marl_agent, single_agent = get_agents()

    state = AgentState()
    state.increment_round()
    marl_intent, marl_path, marl_conf, _ = marl_agent.process(query, state)

    state2 = AgentState()
    state2.increment_round()
    single_intent, single_path, single_conf, _ = single_agent.process(query, state2)

    return {
        "marl": marl_intent, "marl_path": marl_path, "marl_conf": marl_conf,
        "single": single_intent, "single_path": single_path,
        "single_conf": single_conf,
    }


def run_single_query(query: str, expected_intent: str = None):
    """Test a single query with both architectures."""
    print(f"\n{'='*70}")
    print(f"  测试查询: {query}")
    print(f"{'='*70}")

    r = predict_both(query)

    print("\n[MARL 多Agent架构]")
    print(f"  意图: {r['marl'].value}")
    print(f"  路径: {r['marl_path'].value if r['marl_path'] else 'N/A'}")
    print(f"  置信度: {r['marl_conf']:.4f}")

    print("\n[单Agent架构]")
    print(f"  意图: {r['single'].value}")
    print(f"  路径: {r['single_path'].value if r['single_path'] else 'N/A'}")
    print(f"  置信度: {r['single_conf']:.4f}")

    consistent = r["marl"] == r["single"]
    if consistent:
        print(f"\n[结果] OK 两种架构预测一致: {r['marl'].value}")
    else:
        print(f"\n[结果] DIFF 预测不一致: MARL={r['marl'].value}, "
              f"Single={r['single'].value}")

    marl_ok = single_ok = None
    if expected_intent:
        marl_ok = r["marl"].value == expected_intent
        single_ok = r["single"].value == expected_intent
        print(f"  [{'OK' if marl_ok else 'FAIL'}] MARL   实际={r['marl'].value} "
              f"(期望: {expected_intent})")
        print(f"  [{'OK' if single_ok else 'FAIL'}] Single 实际={r['single'].value} "
              f"(期望: {expected_intent})")

    return {
        "query": query,
        "expected": expected_intent,
        "marl": r["marl"].value,
        "single": r["single"].value,
        "marl_ok": marl_ok,
        "single_ok": single_ok,
        "consistent": consistent,
    }


def _failures(results):
    """收集所有失败项：任一架构预测错误都算失败（含两边同时犯错）。"""
    problems = []
    for res in results:
        q, exp = res["query"], res["expected"]
        if res["marl_ok"] is False:
            problems.append(f"[MARL]   {q!r} -> {res['marl']} (期望 {exp})")
        if res["single_ok"] is False:
            problems.append(f"[Single] {q!r} -> {res['single']} (期望 {exp})")
    return problems


def run_all_tests():
    """Run comprehensive tests."""
    print("\n" + "#"*70)
    print("  MARL 系统测试")
    print("#"*70)

    results = [run_single_query(q, expected) for q, expected in TEST_CASES]

    total = len(results)
    marl_correct = sum(1 for r in results if r["marl_ok"])
    single_correct = sum(1 for r in results if r["single_ok"])
    consistent = sum(1 for r in results if r["consistent"])

    print("\n" + "="*70)
    print(f"  MARL   : {marl_correct}/{total} 正确 ({marl_correct/total*100:.1f}%)")
    print(f"  Single : {single_correct}/{total} 正确 ({single_correct/total*100:.1f}%)")
    print(f"  两架构一致: {consistent}/{total}")
    print("="*70)

    problems = _failures(results)
    if problems:
        print("\n  失败项:")
        for p in problems:
            print(f"    {p}")
        print("="*70)
    return not problems


def test_both_architectures_predict_expected():
    """pytest 入口：任一架构预测错误（含两个 agent 同时犯错）都让测试失败。"""
    results = []
    for query, expected in TEST_CASES:
        res = run_single_query(query, expected)
        results.append(res)
    problems = _failures(results)
    assert not problems, "两架构预测回归:\n" + "\n".join(problems)


def test_shared_guard_blocks_write_on_search_query():
    """两架构共用同一护栏：查询类输入一律不得落进写操作意图。"""
    from intent_shared import semantic_conflict_guard, WRITE_INTENTS
    for query in ("帮我搜一下这个文件", "识别截屏中的文字",
                  "提取图片内容", "查找最近创建的文件"):
        for write_intent in WRITE_INTENTS:
            got = semantic_conflict_guard(query, write_intent)
            assert got not in WRITE_INTENTS, \
                f"{query!r} + {write_intent.value} 仍被放行成写意图 {got.value}"


def test_readonly_guard_present_on_both_architectures():
    """只读守卫不能只存在于单体：两套架构的 process 都不得返回写意图。"""
    from intent_shared import WRITE_INTENTS
    marl_agent, single_agent = get_agents()
    query = "只读分析这个文件，不要修改任何东西"
    for name, agent in (("MARL", marl_agent), ("Single", single_agent)):
        state = AgentState()
        state.increment_round()
        intent, _path, _conf, _clar = agent.process(query, state)
        assert intent not in WRITE_INTENTS, \
            f"{name} 的只读守卫未生效，返回了写意图 {intent.value}"


# ---------------------------------------------------------------- 坏例回放
BAD_QUERY = "为我找到异环快捷方式索引的exe文件"


def test_query_subintent_fallback_is_scorer_not_random_ocr():
    """坏例回放：规则 miss + LLM 缺位时，子意图不得随机抽成 query_ocr。

    原缺陷：``marl_agent.get_sub_intent`` 规则未命中后直接
    ``act(explore=True)`` 从 12 类里随机采样，本例被抽中 QUERY_OCR，
    Router 走 OCR 分支、无图即失败 → ``insufficient_evidence`` (-2.0)。
    修后应走 ``query_rewriter.classify_query_subintent`` 确定性计分器，
    与单 Agent 的 LLM→scorer 链路同源。
    """
    from agent_state import QuerySubIntent, RoutingPath

    marl_agent, single_agent = get_agents()
    for name, agent in (("MARL", marl_agent), ("Single", single_agent)):
        state = AgentState()
        state.increment_round()
        intent, path, _conf, _clar = agent.process(BAD_QUERY, state)

        assert intent.value == "query", \
            f"{name} 主意图应为 query，实为 {intent.value}"
        sub = state.current_sub_intent
        assert sub is not None, f"{name} 未给出子意图"
        assert sub != QuerySubIntent.QUERY_OCR, \
            f"{name} 子意图被误判为 query_ocr（会路由到 OCR 并因无图拿 -2.0）"
        assert path != RoutingPath.QUERY_OCR, \
            f"{name} 路径误入 query_ocr"
        assert sub in (QuerySubIntent.QUERY_SEARCH, QuerySubIntent.QUERY_SYSTEM), \
            f"{name} 子意图应为 SEARCH/SYSTEM，实为 {sub.value}"


def test_scorer_and_router_chain_reaches_system_finder():
    """计分器 → 路由映射 → system_finder 三段必须闭环（可断言的链路）。"""
    from agent_state import QuerySubIntent, RoutingPath
    from intent_shared import query_routing_path

    marl_agent, _single = get_agents()
    scored = marl_agent.query_rewriter.classify_query_subintent(BAD_QUERY)
    assert scored in (QuerySubIntent.QUERY_SEARCH, QuerySubIntent.QUERY_SYSTEM), \
        f"计分器应判 SEARCH/SYSTEM，实为 {scored.value}"
    assert query_routing_path(scored) in (RoutingPath.SINGLE_VECTOR,
                                          RoutingPath.QUERY_SYSTEM), \
        f"计分结果映射到了 {query_routing_path(scored).value}（应为可执行查询路径）"
    # 兜底：RULE 词表本身也不得把本例判成 OCR（否则守卫之前就已走偏）
    marl, _ = get_agents()
    assert marl.subintent_agent.rule_predict(BAD_QUERY) != QuerySubIntent.QUERY_OCR


def test_ocr_route_guard_requires_image_evidence():
    """OCR 路由守卫：无图证据必须拦下 QUERY_OCR，真 OCR 查询必须放行。

    守卫是第二道防线（第一道是修 get_sub_intent 的计分器兜底）——
    即使 LLM/神经仍给出 QUERY_OCR，只要没有图像证据就不许进 OCR 分支。
    """
    from router import Router
    from agent_state import SlotType

    marl_agent, _single = get_agents()
    r = Router.__new__(Router)          # 只测守卫，不构造 FileCorpus 等重资产
    r.agent_rl = marl_agent

    # 反例：坏例查询无图片槽位、无 OCR 动词、无图像关键词 → 必须拒绝
    state = AgentState()
    state.slots = {}
    assert not r._ocr_warranted(state, BAD_QUERY), \
        "无图证据的查询竟被放行进 OCR 分支"

    # 正例：三类证据各自都能放行
    for q in ("识别截屏中的文字", "提取图片内容", "OCR一下这张图"):
        assert r._ocr_warranted(AgentState(), q), f"真 OCR 查询 {q!r} 被守卫拦下"
    st = AgentState()
    st.slots = {SlotType.IMAGE_PATH.value: "C:\\tmp\\a.png"}
    assert r._ocr_warranted(st, BAD_QUERY), "显式图片槽位未被守卫认可"


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
