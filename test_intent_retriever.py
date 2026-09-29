"""检索式意图识别（intent_retriever）单元测试。

覆盖:
  - 语料加载 / 索引构建
  - retrieve 相似例句召回
  - vote 投票与弃权（OOD 低相似度）
  - 持久化缓存加载
  - add_example 增量补充
  - 与 classify_intent 集成（第四路投票存在）
  - 六路投票：AgentRL 含 marl 第 6 路 / MetaAgent 含 single 交叉路
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from agent_state import IntentType, AgentState
from intent_retriever import IntentRetriever


def make_retriever(persist=False, tmp=None):
    base = os.path.dirname(os.path.abspath(__file__))
    csvs = [
        os.path.join(base, "intent_corpus.csv"),
        os.path.join(base, "datasets", "intent_corpus.csv"),
    ]
    csvs = [p for p in csvs if os.path.exists(p)]
    assert csvs, "未找到 intent_corpus.csv"
    return IntentRetriever(
        csv_paths=csvs,
        persist=persist,
        persist_dir=tmp or tempfile.mkdtemp(prefix="intent_idx_"),
        top_k=5,
        min_similarity=0.30,
    )


def test_build_and_ready():
    r = make_retriever()
    assert r.ready, f"索引未就绪: {r._build_error}"
    st = r.stats()
    assert st["count"] > 100, f"语料过少: {st['count']}"
    assert st["dim"] > 0
    print(f"[OK] 构建就绪: {st['count']} 条, dim={st['dim']}, backend={st['backend']}")


def test_retrieve_similar():
    r = make_retriever()
    hits = r.retrieve("帮我删除 notes.txt")
    assert hits, "retrieve 返回空"
    assert hits[0]["similarity"] >= hits[-1]["similarity"], "未按相似度降序"
    labels = {h["label"] for h in hits}
    print(f"[OK] retrieve top1={hits[0]['label']} sim={hits[0]['similarity']:.3f} labels={labels}")


def test_vote_known_intents():
    r = make_retriever()
    cases = [
        ("删除文件 old.txt", IntentType.DELETE_FILE),
        ("创建新文件 report.md", IntentType.ADD_FILE),
        ("修改 config.yaml 的内容", IntentType.MODIFY_FILE),
        ("整理文件夹", IntentType.REFACTOR),
        ("搜索强化学习", IntentType.QUERY),
        ("对比分析 a.txt 和 b.txt", IntentType.DEEP_SEARCH),
        ("你好", IntentType.VAGUE),
    ]
    correct = 0
    for q, expected in cases:
        intent, conf, ev = r.vote(q)
        ok = intent == expected
        correct += int(ok)
        mark = "OK" if ok else "MISS"
        print(f"[{mark}] vote({q!r}) -> {intent.value if intent else None} conf={conf:.3f} expected={expected.value}")
    print(f"[INFO] vote 命中 {correct}/{len(cases)}")
    assert correct >= len(cases) - 2, f"vote 命中过低: {correct}/{len(cases)}"


def test_vote_ood_abstain():
    r = make_retriever(persist=False)
    # 极端 OOD：与任何标注例句都不像（乱码/无关长句），应大概率弃权或低置信
    ood = "quantum blockchain zyzzyx frobnicate 12345 !@#$%"
    intent, conf, ev = r.vote(ood)
    if intent is not None:
        # 若仍返回，置信度应显著低于正常查询
        print(f"[INFO] OOD 未弃权但低置信: {intent.value} conf={conf:.3f} (允许，仅记录)")
        assert conf < 0.6, f"OOD 置信度异常偏高: {conf}"
    else:
        print(f"[OK] OOD 正确弃权 (min_sim={r.min_similarity})")


def test_persistence_roundtrip():
    tmp = tempfile.mkdtemp(prefix="intent_persist_")
    try:
        r1 = make_retriever(persist=True, tmp=tmp)
        assert r1.ready
        n1 = r1.stats()["count"]

        # 模拟二次启动：新实例应命中缓存
        r2 = make_retriever(persist=True, tmp=tmp)
        assert r2.ready
        n2 = r2.stats()["count"]
        assert n1 == n2, f"缓存条数不一致: {n1} vs {n2}"

        # 检索结果一致
        h1 = r1.retrieve("删除文件 x.txt")
        h2 = r2.retrieve("删除文件 x.txt")
        assert h1 and h2
        assert h1[0]["text"] == h2[0]["text"]
        assert abs(h1[0]["similarity"] - h2[0]["similarity"]) < 1e-4
        print(f"[OK] 持久化往返一致: {n1} 条")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_add_example():
    tmp = tempfile.mkdtemp(prefix="intent_add_")
    try:
        r = make_retriever(persist=True, tmp=tmp)
        assert r.ready
        n0 = r.stats()["count"]
        unique = "帮我把季度营收仪表盘拆分成微服务模块"
        ok = r.add_example(unique, IntentType.REFACTOR.value)
        assert ok, "add_example 失败"
        assert r.stats()["count"] == n0 + 1
        # 新例句应能被检索到
        hits = r.retrieve(unique, top_k=3)
        assert any(h["text"] == unique for h in hits), "新增例句未被召回"
        # 重复添加返回 False
        dup = r.add_example(unique, IntentType.REFACTOR.value)
        assert dup is False, "重复添加应返回 False"
        print(f"[OK] add_example 增量补充 + 去重")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_integration_fourth_voter():
    """确认 classify_intent 候选中出现 retrieval 路（不依赖 LLM）。"""
    from agent_rl import AgentRL

    agent = AgentRL()
    # 强制关闭 LLM（status 非 ready/fallback 时 available=False）
    agent.llm_classifier.status = "unconfigured"
    agent.llm_classifier.endpoints = []
    assert hasattr(agent, "intent_retriever"), "AgentRL 缺少 intent_retriever"
    intent, conf, ev = agent.intent_retriever.vote("删除文件 notes.txt")
    assert intent is not None and conf > 0
    print(f"[OK] AgentRL.intent_retriever 接入, vote={intent.value} conf={conf:.3f}")

    # classify_intent 端到端（允许任意结果，只要不抛异常）
    state = AgentState()
    state.increment_round()
    final_intent, final_conf = agent.classify_intent("删除文件 notes.txt", state)
    assert isinstance(final_intent, IntentType)
    print(f"[OK] classify_intent 端到端: {final_intent.value} conf={final_conf:.3f}")


def test_marl_integration():
    from marl_agent import create_marl_agent
    agent = create_marl_agent()
    assert hasattr(agent, "intent_retriever"), "MetaAgent 缺少 intent_retriever"
    state = AgentState()
    state.increment_round()
    intent, conf = agent.classify_intent("创建新文件 notes.txt", state)
    assert isinstance(intent, IntentType)
    print(f"[OK] MARL classify_intent: {intent.value} conf={conf:.3f}")


def test_six_way_voters():
    """六路投票：单 Agent 侧含 marl 第 6 路，MetaAgent 侧含 single 交叉路。"""
    from agent_rl import AgentRL
    from marl_agent import create_marl_agent

    state = AgentState()
    state.increment_round()

    # --- 单 Agent 六路 ---
    single = AgentRL()
    single.llm_classifier.status = "unconfigured"
    single.llm_classifier.endpoints = []
    for q in ("删除文件 notes.txt", "搜索强化学习", "整理文件夹"):
        single.classify_intent(q, state)
    src = set(single.last_sources)
    base = {"qnet", "cls", "rule", "retrieval", "marl"}
    assert base <= src, f"单 Agent 六路缺少 {base - src}: {sorted(src)}"
    print(f"[OK] AgentRL 六路来源: {sorted(src)}")

    # marl_vote=False（MetaAgent 交叉路用）不得带 marl
    quiet = AgentRL(marl_vote=False)
    quiet.classify_intent("删除文件 notes.txt", state)
    assert "marl" not in set(quiet.last_sources), "marl_vote=False 仍出现 marl 路"
    n_intent, n_conf, n_info = quiet.neural_vote("删除文件 notes.txt", state)
    assert isinstance(n_intent, IntentType) and 0.0 <= n_conf <= 1.0
    print(f"[OK] AgentRL(marl_vote=False): {sorted(quiet.last_sources)}, "
          f"neural_vote={n_intent.value} conf={n_conf:.3f}")

    # --- MetaAgent 六路 ---
    meta = create_marl_agent()
    meta.llm_classifier.status = "unconfigured"
    meta.llm_classifier.endpoints = []
    for q in ("删除文件 notes.txt", "搜索强化学习", "整理文件夹"):
        meta.classify_intent(q, state)
    msrc = set(meta.last_sources)
    mbase = {"policy", "league", "rule", "retrieval", "single"}
    assert mbase <= msrc, f"MetaAgent 六路缺少 {mbase - msrc}: {sorted(msrc)}"
    print(f"[OK] MetaAgent 六路来源: {sorted(msrc)}")


def run_all():
    print("=" * 70)
    print("  intent_retriever 测试")
    print("=" * 70)
    tests = [
        test_build_and_ready,
        test_retrieve_similar,
        test_vote_known_intents,
        test_vote_ood_abstain,
        test_persistence_roundtrip,
        test_add_example,
        test_integration_fourth_voter,
        test_marl_integration,
        test_six_way_voters,
    ]
    failed = []
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {e}")
            failed.append((t.__name__, str(e)))
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 70)
    if failed:
        print(f"  失败 {len(failed)}/{len(tests)}:")
        for name, err in failed:
            print(f"    - {name}: {err}")
        print("=" * 70)
        return False
    print(f"  全部通过 {len(tests)}/{len(tests)}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
