"""intent_shared 共享投票/守卫语义单元测试。

覆盖问题③（同源同错）中「投票聚合」的修复：
  - 旧实现 ``score = Σ/n`` 是**共识惩罚**：两路高置信同向支持会被一路略高的
    单源意图击败；
  - 新实现 ``score = Σ/n**α``（α = AGG_DAMPING_EXP）是**共识增强**，
    且 n = 1 时与旧实现严格相等（不改动历史阈值标度）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent_state import IntentType
import intent_shared


def _score(intent, scores):
    """scores: [(分, 来源名), ...] → 聚合后该意图的共识得分。"""
    votes = intent_shared.aggregate_votes(
        [(intent, s, name) for s, name in scores])
    return votes[intent]["score"]


def test_single_source_score_unchanged():
    """单源时得分必须等于原始得分：CONFIDENCE_THRESHOLD(0.65)/0.4 的标度不变。"""
    for raw in (0.0, 0.3, 0.65, 1.0):
        got = _score(IntentType.QUERY, [(raw, "rule")])
        assert abs(got - raw) < 1e-12, f"单源得分被改动: {raw} -> {got}"
    print("[OK] 单源得分与原始分严格一致（阈值标度不变）")


def test_consensus_beats_dilution_defect():
    """共识惩罚缺陷：0.90+0.80 必须反超一路 0.86。"""
    two = _score(IntentType.QUERY, [(0.90, "rule"), (0.80, "retrieval")])
    one = _score(IntentType.DEEP_SEARCH, [(0.86, "rule")])
    assert two > one, f"共识仍被惩罚: 两路 {two:.4f} <= 单路 {one:.4f}"
    print(f"[OK] 共识增强: 两路 {two:.4f} > 单路 {one:.4f}")


def test_weak_consensus_cannot_outrank_strong_single():
    """弱票堆积不得淹没强证据：5 路 0.20 不能压过一路 0.85。"""
    weak = _score(IntentType.REPORT_GENERATE, [(0.20, "qnet")] * 5)
    strong = _score(IntentType.DEEP_SEARCH, [(0.85, "rule")])
    assert weak < strong, f"弱共识淹没强证据: {weak:.4f} >= {strong:.4f}"
    print(f"[OK] 弱共识不越界: {weak:.4f} < {strong:.4f}")


def test_confidence_bounded_and_monotonic_in_consensus():
    """置信度 ∈ [0,1]，且在质量不变时随同向支持路数单调不降。"""
    for n in (1, 2, 3, 4, 6):
        votes = intent_shared.aggregate_votes(
            [(IntentType.QUERY, 0.7, f"s{i}") for i in range(n)])
        conf = intent_shared.vote_confidence(votes, IntentType.QUERY)
        assert 0.0 <= conf <= 1.0, f"置信度越界: n={n} conf={conf}"
        prev = intent_shared.vote_confidence(
            intent_shared.aggregate_votes(
                [(IntentType.QUERY, 0.7, f"s{i}") for i in range(n - 1)]),
            IntentType.QUERY) if n > 1 else 0.0
        assert conf >= prev - 1e-12, f"共识反而降分: n={n} {conf} < {prev}"
    print("[OK] 置信度有界且随共识路数单调不降")


def test_confidence_zero_for_unknown_intent():
    votes = intent_shared.aggregate_votes(
        [(IntentType.QUERY, 0.9, "rule")])
    assert intent_shared.vote_confidence(votes, IntentType.VAGUE) == 0.0
    print("[OK] 未获支持的意图置信度为 0")


def test_both_architectures_use_the_same_aggregation():
    """两套架构必须共用同一个聚合函数对象（问题①：逐字复制）。"""
    import agent_rl
    import marl_agent

    assert agent_rl.aggregate_votes is intent_shared.aggregate_votes, \
        "AgentRL 未使用共享 aggregate_votes"
    assert marl_agent.aggregate_votes is intent_shared.aggregate_votes, \
        "MetaAgent 未使用共享 aggregate_votes"
    assert agent_rl._shared_vote_confidence is intent_shared.vote_confidence
    assert marl_agent._shared_vote_confidence is intent_shared.vote_confidence
    print("[OK] 两架构共用同一份 aggregate_votes / vote_confidence")


def test_damping_exponent_is_documented_value():
    assert intent_shared.AGG_DAMPING_EXP == 0.75, \
        f"聚合指数被改动: {intent_shared.AGG_DAMPING_EXP}"
    print("[OK] AGG_DAMPING_EXP = 0.75")


# ================================================================
# 问题① 逐字复制：常量/模板/奖励必须只有唯一定义
# ================================================================
def test_templates_and_thresholds_are_single_copy():
    import agent_rl
    import marl_agent
    import train_strong

    assert agent_rl.AgentRL.TASK_TEMPLATES is intent_shared.TASK_TEMPLATES, \
        "AgentRL.TASK_TEMPLATES 是独立副本"
    assert marl_agent.TASK_TEMPLATES is intent_shared.TASK_TEMPLATES, \
        "marl_agent.TASK_TEMPLATES 是独立副本"
    assert train_strong.eval_pool is intent_shared.eval_pool, \
        "train_strong 用的不是共享 eval_pool"
    for name in ("SHARED_CONFIDENCE_THRESHOLD", "SHARED_ENGRAM_MIN_MARGIN"):
        for mod in (agent_rl, marl_agent):
            assert getattr(mod, name, None) == getattr(intent_shared, name.replace("SHARED_", "")), \
                f"{mod.__name__}.{name} 与 intent_shared 漂移"
    assert agent_rl.AgentRL.CONFIDENCE_THRESHOLD == intent_shared.CONFIDENCE_THRESHOLD
    meta = getattr(marl_agent, "MetaAgent", None)
    if meta is not None and hasattr(meta, "CONFIDENCE_THRESHOLD"):
        assert meta.CONFIDENCE_THRESHOLD == intent_shared.CONFIDENCE_THRESHOLD
    print("[OK] TASK_TEMPLATES / 阈值只有唯一定义")


def test_train_and_eval_pools_are_disjoint():
    """问题③：训练集=测试集会让两侧准确率一起虚高。"""
    checked = 0
    for intent, templates in intent_shared.TASK_TEMPLATES.items():
        if len(templates) < 2:
            continue
        train = set(intent_shared.train_pool(templates))
        held_out = set(intent_shared.eval_pool(templates))
        assert train and held_out, f"{intent.value}: 留出池为空"
        assert not (train & held_out), \
            f"{intent.value}: 训练池与留出池有交集 {train & held_out}"
        assert (train | held_out) == set(templates), f"{intent.value}: 切分丢失模板"
        checked += 1
    assert checked >= 5, f"可校验的意图过少: {checked}"
    print(f"[OK] 训练池/留出池无交集且并集完整（{checked} 个意图）")


def test_train_reward_is_shared_and_unified():
    """问题②：单体答对 15、MARL 答对 5，跨架构 reward 曲线差 3 倍。"""
    import agent_rl
    import marl_agent
    import train_strong

    assert agent_rl._shared_train_reward is intent_shared.train_reward
    assert marl_agent._shared_train_reward is intent_shared.train_reward
    assert train_strong.train_reward is intent_shared.train_reward
    assert intent_shared.train_reward(True) == 15, intent_shared.train_reward(True)
    assert intent_shared.train_reward(False) == -2, intent_shared.train_reward(False)
    print("[OK] train_reward 唯一定义：答对 15 / 答错 -2")


# ================================================================
# 问题③ 机制缺陷：单步自博弈不得用 next_state bootstrap 自己
# ================================================================
def test_agentrl_single_step_training_marks_done():
    import agent_rl

    agent = agent_rl.AgentRL()
    agent.epsilon = 0.0

    seen = {"done": [], "reward": [], "train_pool": []}
    orig_store = agent.store_experience
    orig_train_pool = agent_rl._shared_train_pool

    def spy_store(sa, action, reward, ns, done):
        seen["done"].append(bool(done))
        seen["reward"].append(float(reward))
        return orig_store(sa, action, reward, ns, done)

    def spy_pool(templates):
        pool = orig_train_pool(templates)
        seen["train_pool"].append(pool)
        return pool

    agent.store_experience = spy_store
    agent_rl._shared_train_pool = spy_pool
    try:
        agent.train_episode(num_episodes=1)
    finally:
        agent.store_experience = orig_store
        agent_rl._shared_train_pool = orig_train_pool

    assert seen["done"], "train_episode 没有写入任何经验"
    assert all(seen["done"]), \
        "存在 done=False 的单步自举样本：next_state == state 会让 TD 退化"
    bad = [r for r in seen["reward"] if r not in (15.0, -2.0)]
    assert not bad, f"train_episode 使用了共享之外的奖励值: {set(bad)}"
    assert seen["train_pool"], "train_episode 未走 train_pool 切分"
    print(f"[OK] train_episode: {len(seen['done'])} 条经验全部 done=True，"
          f"奖励取值 {sorted(set(seen['reward']))}")


def test_evaluate_uses_holdout_pool():
    """evaluate 必须读留出池，不能和训练池共用模板。"""
    import agent_rl

    agent = agent_rl.AgentRL()
    held_out = []
    orig = agent_rl._shared_eval_pool

    def spy(templates):
        pool = orig(templates)
        held_out.append(set(pool))
        return pool

    agent_rl._shared_eval_pool = spy
    try:
        agent.evaluate()
    finally:
        agent_rl._shared_eval_pool = orig

    assert held_out, "evaluate 未调用 eval_pool —— 仍在用训练模板自测"
    overlapping = []
    for intent, templates in intent_shared.TASK_TEMPLATES.items():
        train = set(intent_shared.train_pool(templates))
        for pool in held_out:
            if pool & train:
                overlapping.append((intent.value, pool & train))
    assert not overlapping, f"评估池与训练池重叠: {overlapping[:3]}"
    print(f"[OK] evaluate 走 eval_pool（{len(held_out)} 次），与训练池零重叠")


# ================================================================
# 问题② 漂移：问候语判定必须让位给明确的写操作指令
# ================================================================
def test_safe_fallback_puts_write_instructions_first():
    import agent_rl

    agent = agent_rl.AgentRL()
    cases = [
        ("添加文件 notes.txt 内容为 你好", "add_file"),
        ("删除 old.txt 谢谢", "delete_file"),
        ("修改 config.yaml 嗯", "modify_file"),
    ]
    for query, expected in cases:
        got = agent._safe_fallback(query)
        assert got.value == expected, \
            f"{query!r} 被回退成 {got.value}，问候词抢在了写操作之前"
    print(f"[OK] _safe_fallback 问候词不再压过写操作（{len(cases)} 例）")


# ================================================================
# 问题④ 错误传播链路：失败样本必须带架构归属、回流不支持要显式报错
# ================================================================
def test_badcase_replay_rejects_unsupported_agent_explicitly():
    import badcase_replay as br

    ok_none, reason_none = br.replay_capability(None)
    assert not ok_none and reason_none, "agent=None 未给出原因"

    from marl_agent import create_marl_agent
    ok_m, reason_m = br.replay_capability(create_marl_agent())
    assert not ok_m and reason_m, \
        "MARL 回流能力未被判定为不支持（旧实现会逐条 except 静默吞掉）"
    pushed, skipped = br.push_bad_cases(create_marl_agent(), dry_run=True)
    assert pushed == 0 and skipped == 0, \
        f"不支持的 agent 竟然回流了: pushed={pushed} skipped={skipped}"

    from agent_rl import AgentRL
    ok_s, reason_s = br.replay_capability(AgentRL())
    assert ok_s, f"单体 AgentRL 应支持回流: {reason_s}"
    print(f"[OK] 回流能力显式判定：single=支持 / marl=拒绝（{reason_m[:40]}…）")


def test_rule_tables_are_derived_from_shared():
    """问题①/②：两套架构的规则关键词表必须从 intent_shared 的表拼出来。"""
    from agent_state import IntentType
    from marl_agent import create_marl_agent

    rk = create_marl_agent().intent_agent.rule_keywords
    assert tuple(rk[IntentType.QUERY]) == intent_shared.SEARCH_KEYWORDS, \
        "MetaAgent QUERY 关键词与共享表不一致"
    assert tuple(rk[IntentType.DELETE_FILE]) == intent_shared.DELETE_KEYWORDS
    assert tuple(rk[IntentType.ADD_FILE]) == intent_shared.ADD_FILE_KEYWORDS
    assert tuple(rk[IntentType.MODIFY_FILE]) == intent_shared.MODIFY_FILE_KEYWORDS
    assert tuple(rk[IntentType.REFACTOR]) == intent_shared.REFACTOR_KEYWORDS
    assert tuple(rk[IntentType.DEEP_SEARCH]) == intent_shared.DEEP_ANALYSIS_KEYWORDS
    # 这三张是「共享表 + 一个专属词」，共享表必须是前缀（前 N 项逐字相等）
    for intent, shared in (
        (IntentType.FILE_ANALYSIS, intent_shared.ANALYSIS_KEYWORDS),
        (IntentType.FILE_SCAN, intent_shared.SCAN_KEYWORDS),
        (IntentType.REPORT_GENERATE, intent_shared.REPORT_KEYWORDS),
    ):
        got = tuple(rk[intent][:len(shared)])
        assert got == shared, f"{intent.value} 未从共享表派生: {got} != {shared}"

    import inspect
    import agent_rl
    src = inspect.getsource(agent_rl.AgentRL._safe_fallback)
    for name in ("ANALYSIS_GATE_KEYWORDS", "FORBIDDEN_MODIFY_KEYWORDS",
                 "SCAN_KEYWORDS", "REPORT_KEYWORDS", "SEARCH_KEYWORDS"):
        assert name in src, f"_safe_fallback 仍在用内联词表（缺少 {name}）"
    print("[OK] 规则表 / 回退闸门全部从共享表派生")


def test_trace_and_failure_records_carry_agent_attribution():
    import dataclasses
    import inspect

    import continuous_learning as cl

    cls = next(c for c in vars(cl).values()
               if inspect.isclass(c) and hasattr(c, "record_trace"))
    assert "agent" in {f.name for f in dataclasses.fields(cl.TraceRecord)}, \
        "TraceRecord 缺少 agent 字段"
    assert "agent" in inspect.signature(cls.record_trace).parameters, \
        "record_trace 不接受 agent="
    src = inspect.getsource(cls._record_failure)
    assert '"agent"' in src, "_record_failure 没有把架构写进失败样本"

    base = os.path.dirname(os.path.abspath(__file__))
    for fname, needle in (
        ("main.py", "agent=self.agent_type"),
        ("chat_server.py", "agent=getattr(system, \"agent_type\""),
    ):
        with open(os.path.join(base, fname), encoding="utf-8") as fh:
            text = fh.read()
        assert needle in text, f"{fname} 未把 agent_type 传给 record_trace"
    print("[OK] 失败样本带架构归属（TraceRecord / _record_failure / 两个入口）")


def test_intent_label_is_a_single_implementation():
    """意图标签只能有一份实现（原 agent_rl/marl 各一份逐字 one-liner）。"""
    from agent_state import SHORT_LABELS, AgentState
    for intent in IntentType:
        want = SHORT_LABELS.get(intent, intent.value)
        got = intent_shared.intent_label(intent)
        assert got == want, f"{intent} 标签漂移: {got} != {want}"

    base = os.path.dirname(os.path.abspath(__file__))
    for fname in ("agent_rl.py", "marl_agent.py"):
        with open(os.path.join(base, fname), encoding="utf-8") as fh:
            text = fh.read()
        assert "SHORT_LABELS.get(" not in text, \
            f"{fname} 仍自己查标签表（应委托 intent_shared.intent_label）"
        assert "return intent_label(intent)" in text, \
            f"{fname}.get_intent_label 没有委托共享实现"
    print("[OK] 意图标签单一实现，两架构均委托 intent_shared.intent_label")


def test_state_vector_success_count_not_double_counted():
    """agent_state.py:254 的回归护栏。

    旧 ``get_state_vector`` 把 ``success_total`` 算成
    ``successful + added + modified + refactored + deleted`` —— 五个计数器
    指向同一批成功操作，等于把同一件事重复统计 5 次（第 2 维饱和到 0.5）。
    """
    from agent_state import AgentState
    st = AgentState()
    for kind in ("add", "modify", "refactor", "delete"):
        st.record_operation(kind, "f.py", True, "ok")
    assert st.successful_ops == 4, st.successful_ops
    vec = st.get_state_vector()
    assert abs(vec[1] - st.successful_ops / 20.0) < 1e-9, \
        f"第 2 维应为 successful_ops/20，实为 {vec[1]}"

    naive = (st.successful_ops + st.added_count + st.modified_count
             + st.refactored_count + st.deleted_count) / 20.0
    assert abs(vec[1] - naive) > 1e-9, "第 2 维仍在重复计数（bug 回归）"
    print("[OK] get_state_vector 不再重复计数成功操作")


def test_route_maps_cover_every_member():
    """两张路由表必须覆盖全部枚举成员，否则就掉进 .get 兜底。

    旧实现 ``route_intent`` 用 ``routing_map[intent]`` 硬索引，
    ``query_routing_path`` 用 ``.get(...)`` —— 两处口径不一致，
    将来给 IntentType 加新成员会在线上直接 KeyError。现在两表都走
    ``.get``，但**覆盖缺口必须在测试里显式失败**，不能被兜底悄悄吞掉。
    """
    from agent_state import QuerySubIntent, RoutingPath

    missing = [m.name for m in IntentType if m not in intent_shared.ROUTING_MAP]
    assert not missing, f"IntentType 成员没有登记路由（会被兜底吞掉）: {missing}"

    missing_sub = [m.name for m in QuerySubIntent
                   if m not in intent_shared.QUERY_ROUTING_MAP]
    assert not missing_sub, f"QuerySubIntent 成员没有登记路由: {missing_sub}"

    for m in IntentType:
        assert isinstance(intent_shared.route_intent(m), RoutingPath)
    print("[OK] 路由表覆盖全部枚举成员（新增成员会在 CI 上被点名）")


def run_all_tests():
    print("\n" + "#" * 70)
    print("  intent_shared 共享语义测试")
    print("#" * 70)
    test_single_source_score_unchanged()
    test_consensus_beats_dilution_defect()
    test_weak_consensus_cannot_outrank_strong_single()
    test_confidence_bounded_and_monotonic_in_consensus()
    test_confidence_zero_for_unknown_intent()
    test_both_architectures_use_the_same_aggregation()
    test_damping_exponent_is_documented_value()
    test_templates_and_thresholds_are_single_copy()
    test_train_and_eval_pools_are_disjoint()
    test_train_reward_is_shared_and_unified()
    test_agentrl_single_step_training_marks_done()
    test_evaluate_uses_holdout_pool()
    test_safe_fallback_puts_write_instructions_first()
    test_badcase_replay_rejects_unsupported_agent_explicitly()
    test_rule_tables_are_derived_from_shared()
    test_trace_and_failure_records_carry_agent_attribution()
    test_intent_label_is_a_single_implementation()
    test_state_vector_success_count_not_double_counted()
    test_route_maps_cover_every_member()
    return True


if __name__ == "__main__":
    run_all_tests()
