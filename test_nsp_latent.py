"""NSP 潜空间单元测试 —— 覆盖评审提出的五个缺口的落点。

  1. 三专家（MARL）接上 NSP、且实现只有单副本
  2. 目标不是「自己预测自己」：EMA 冻结副本 + InfoNCE
  3. 不止一步：H 步自由运行 rollout + 想象 Q 混入 TD bootstrap
  4. 不是确定性点估计：(μ, logσ²) + 高斯 NLL + KL
  5. 有可复现的评估指标：probe() 固定输入 → 固定输出

不变量（这些一旦破了，回归 gate / 旧 checkpoint 就会静默坏掉）：
  - ``build_nsp`` 不扰动全局随机流
  - NSP 不改 ``q_network`` 的 state_dict 结构（strict load）
  - ``nsp.enabled=False`` 时想象分量与梯度都归零

用法::

    python test_nsp_latent.py
    pytest test_nsp_latent.py
"""

import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

TMP_ROOT = tempfile.mkdtemp(prefix="agentrl_nsp_")

# 惰性单例：构造 AgentRL / MetaAgent 各要几秒，多个用例共用一份
_AGENTS = {}


def single_agent():
    if "single" not in _AGENTS:
        from agent_rl import AgentRL
        _AGENTS["single"] = AgentRL(marl_vote=False)
    return _AGENTS["single"]


def marl_agent():
    if "marl" not in _AGENTS:
        from marl_agent import create_marl_agent
        _AGENTS["marl"] = create_marl_agent()
    return _AGENTS["marl"]


def _fresh_nsp(task_aware: bool = True):
    from transformer_arch import TransformerDuelingQNetworkWithNSP, LoopedTransformer
    from nsp_latent import build_nsp

    if task_aware:
        net = TransformerDuelingQNetworkWithNSP(state_dim=268, d_model=128,
                                                num_actions=11)
        return build_nsp(net.transformer, net.pool, action_dim=11, d_model=128,
                         num_actions=11, nsp_task=1)
    enc = LoopedTransformer(128, num_loops=2, state_dim=272)
    return build_nsp(enc, None, action_dim=64, d_model=128, num_actions=64)


# ================================================================ 构造不污染 RNG
def test_build_nsp_does_not_perturb_global_rng():
    from transformer_arch import LoopedTransformer
    from nsp_latent import build_nsp

    # 基准：只构造编码器之后的随机流
    torch.manual_seed(1234)
    LoopedTransformer(128, num_loops=2, state_dim=272)
    expected = torch.randn(4)

    # 实验：同样构造编码器，再多构造一次 NSP
    torch.manual_seed(1234)
    enc = LoopedTransformer(128, num_loops=2, state_dim=272)
    build_nsp(enc, None, action_dim=8, d_model=128, num_actions=8, seed=7)
    got = torch.randn(4)

    assert torch.equal(expected, got), "build_nsp 吃掉了全局随机流，回归 gate 会跟着错位"


def test_online_encoder_is_not_registered_as_submodule():
    nsp = _fresh_nsp()
    online = {id(p) for p in nsp.online_encoder.parameters()}
    registered = {id(p) for p in nsp.parameters()}
    assert not (online & registered), \
        "在线编码器被注册成子模块 → 宿主网络参数会在优化器里出现两次"
    assert any(k.startswith("target.") for k in nsp.state_dict()), \
        "目标编码器必须进 state_dict（否则 checkpoint 里没有它）"
    assert not any(k.startswith("_online") for k in nsp.state_dict()), \
        "在线编码器不得重复写进 state_dict"


def test_optimizer_has_no_duplicate_parameters():
    agent = single_agent()
    ids = [id(p) for group in agent.optimizer.param_groups
           for p in group["params"]]
    assert len(ids) == len(set(ids)), "优化器里有重复参数"


# ================================================================ EMA 目标
def test_target_is_frozen_and_tracks_online():
    nsp = _fresh_nsp()
    assert all(not p.requires_grad for p in nsp.target.parameters()), \
        "目标编码器必须 stop-gradient"

    with torch.no_grad():
        for p in nsp.online_encoder.parameters():
            p.add_(1.0)

    drift_before = nsp.ema_drift()
    assert drift_before > 1e-6, "在线被改动后目标应当产生漂移"

    nsp.update_ema(tau=1.0)
    assert nsp.ema_drift() < 1e-6, "tau=1.0 的 EMA 应当一次性贴齐在线编码器"
    assert nsp.ema_drift() < drift_before


def test_refresh_target_snaps_after_reinit():
    nsp = _fresh_nsp()
    with torch.no_grad():
        for p in nsp.online_encoder.parameters():
            p.mul_(0.0).add_(3.0)
    nsp.refresh_target()
    assert nsp.ema_drift() < 1e-6, "refresh_target 用于宿主网络重新初始化之后"


# ================================================================ 分布式转移头
def test_transition_head_outputs_mu_and_logvar():
    from nsp_latent import StochasticTransitionHead, LOGVAR_MIN, LOGVAR_MAX

    head = StochasticTransitionHead(d_model=16, action_dim=5)
    z = torch.randn(7, 16)
    actions = torch.randint(0, 5, (7,))
    mu, logvar = head(z, actions)

    assert mu.shape == (7, 16) and logvar.shape == (7, 16)
    assert torch.all(logvar >= LOGVAR_MIN) and torch.all(logvar <= LOGVAR_MAX)
    # 末层零初始化：起点必须是「温和的单位高斯」，不能一上来就发散
    assert float(mu.abs().max()) < 1e-6
    assert float((logvar - 0.0).abs().max()) < 1e-6

    # 采样出来的不是点估计
    s1 = StochasticTransitionHead.sample(mu, logvar)
    s2 = StochasticTransitionHead.sample(mu, logvar)
    assert not torch.allclose(s1, s2), "同一 (μ,σ) 两次采样应给出不同样本"
    assert torch.allclose(StochasticTransitionHead.sample(mu, logvar, True), mu)


def test_supervise_uses_nll_and_kl_not_pure_mse():
    nsp = _fresh_nsp()
    # 末层零初始化 ⇒ 初始 μ≡0、logσ²≡0 ⇒ KL 恰为 0。给末层一点权重，
    # 才能验证 KL 这一项真的在算（纯 MSE 实现里它是恒 0 的死项）。
    with torch.no_grad():
        nsp.transition.net[-1].weight.normal_(0.0, 0.05)
        nsp.transition.net[-1].bias.normal_(0.0, 0.05)

    b = 16
    states = torch.randn(b, 268)
    next_states = torch.randn(b, 268)
    actions = torch.randint(0, 11, (b,))

    stats = nsp.supervise(states, actions, next_states, torch.randn(b, 11))
    for key in ("nsp_nll", "nsp_kl", "nsp_infonce", "nsp_infonce_acc",
                "nsp_bridge_mse", "nsp_total"):
        assert key in stats, f"缺指标 {key}"
        assert torch.isfinite(stats[key]), f"{key} 非有限值"
    assert float(stats["nsp_kl"]) > 0.0, "KL(N(0,1)) 对非零 μ/σ² 必须为正"
    assert 0.0 <= float(stats["nsp_infonce_acc"]) <= 1.0

    # total = nll + β·kl + InfoNCE + 桥标定，权重都在
    expected = (float(stats["nsp_nll"]) + nsp.beta_kl * float(stats["nsp_kl"])
                + nsp.infonce_weight * float(stats["nsp_infonce"])
                + nsp.bridge_weight * float(stats["nsp_bridge_mse"]))
    assert abs(expected - float(stats["nsp_total"])) < 1e-4


def test_infonce_negatives_prevent_collapse():
    """把 q 塌成同一行时 InfoNCE 必须涨到 ≈log(batch)，batch 内负样本在起作用。"""
    from nsp_latent import NSPLatentModule  # noqa: F401  （确保类已加载）
    import torch.nn.functional as F

    nsp = _fresh_nsp()
    torch.manual_seed(0)
    k = F.normalize(torch.randn(16, 8), dim=-1)
    aligned = float(nsp._infonce(k, k)[0])
    collapsed = float(nsp._infonce(k[:1].expand(16, -1), k)[0])

    assert aligned < math.log(16) - 0.5, f"完美对齐时应很低，实际 {aligned:.3f}"
    assert collapsed > aligned, \
        f"塌缩后 InfoNCE 必须变大（{aligned:.3f} → {collapsed:.3f}）"
    assert collapsed > math.log(16) - 0.5


# ================================================================ 想象 Q 混入 TD
def test_imagine_q_target_alpha_gate_and_clamp():
    nsp = _fresh_nsp()
    b = 12
    states = torch.randn(b, 268)
    actions = torch.randint(0, 11, (b,))
    q_next = torch.randn(b)

    unchanged, _ = nsp.imagine_q_target(states, actions, q_next, alpha=0.0)
    assert torch.equal(unchanged, q_next), "α=0 必须逐位等价于改造前的 TD 目标"

    mixed, imag = nsp.imagine_q_target(states, actions, q_next, alpha=1.0,
                                       clamp=0.5)
    assert not torch.allclose(mixed, q_next)
    assert torch.all((mixed - q_next).abs() <= 0.5 + 1e-6), \
        "想象分量必须被 clamp 限制，否则早期桥未收敛会污染 TD"
    assert imag.shape == q_next.shape


def test_imagine_alpha_ramps_from_zero():
    from agent_rl import AgentRL

    a = AgentRL.__new__(AgentRL)
    a.nsp_imagine_alpha = 0.2
    a.nsp_imagine_warmup = 1000

    class _Nsp:
        enabled = True

    a.nsp = _Nsp()
    a.training_step = 0
    assert a._imagine_alpha() == 0.0, "第 0 步必须是 0"
    a.training_step = 500
    assert 0.0 < a._imagine_alpha() < 0.2
    a.training_step = 5000
    assert abs(a._imagine_alpha() - 0.2) < 1e-9
    a.nsp.enabled = False
    assert a._imagine_alpha() == 0.0, "消融后想象分量必须恒为 0"


# ================================================================ H 步轨迹
def test_segments_do_not_cross_done_or_seg_boundary():
    from agent_rl import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(capacity=64)
    d = 4
    # 三段轨迹：3 步 / 2 步 / 4 步，中间用 done 切开
    for length, done_at_end in ((3, True), (2, True), (4, True)):
        for k in range(length):
            s = np.full(d, k + 1, dtype=np.float32)
            buf.push(s, k % 5, 1.0, s + 1, done=(k == length - 1))

    found = 0
    for _ in range(200):
        for seg in buf.sample_segments(4, 2):
            found += 1
            assert seg["states"].shape == (3, d)
            assert seg["actions"].shape == (2,)
            assert seg["next_states"].shape == (3, d)
            # 中间不得有 done，否则那不是一条连续轨迹
            assert not seg["dones"][:-1].any(), seg["dones"]
            # 同一段内状态必须真的连续（k → k+1）
            diffs = np.diff(seg["states"], axis=0)
            assert np.allclose(diffs, 1.0), diffs
    assert found > 0, "3 步轨迹应该能被采成 H=2 的片段"

    # 全是单步 episode（单 Agent 自博弈就是这样）→ 不该编造多步轨迹
    solo = PrioritizedReplayBuffer(capacity=32)
    for k in range(16):
        s = np.full(d, k, dtype=np.float32)
        solo.push(s, 0, 1.0, s, done=True)
    assert solo.sample_segments(4, 2) == [], "单步 episode 不得产出 H 步片段"


def test_segment_losses_produces_finite_rollout_error():
    nsp = _fresh_nsp()
    g, h, d = 4, 3, 268
    states = torch.randn(g, h + 1, d)
    actions = torch.randint(0, 11, (g, h))
    next_states = torch.randn(g, h + 1, d)

    stats = nsp.segment_losses(states, actions, next_states)
    assert "nsp_total" in stats and "nsp_rollout_err" in stats
    assert torch.isfinite(stats["nsp_total"])
    assert torch.isfinite(stats["nsp_rollout_err"])
    assert float(stats["nsp_rollout_err"]) >= 0.0

    # 形状不对就返回空，而不是崩溃或造数据
    assert nsp.segment_losses(states, actions[:, :0], next_states) == {}, \
        "零步动作序列必须返回空"
    assert nsp.segment_losses(states[0], actions, next_states) == {}, \
        "丢掉 batch 维后必须返回空"


# ================================================================ 动作离散化
def test_discretize_actions_handles_multi_discrete():
    from nsp_latent import discretize_actions

    a = torch.tensor([0, 1, 10, 11, 11], dtype=torch.int64)
    out = discretize_actions(a, 11)
    assert out.dtype == torch.int64
    assert int(out.min()) >= 0 and int(out.max()) < 11

    multi = torch.randint(0, 7, (5, 5), dtype=torch.int64)
    code = discretize_actions(multi, 64)
    assert code.shape == (5,)
    assert int(code.min()) >= 0 and int(code.max()) < 64
    # 同一动作序列必须得到同一个码（确定性）
    assert torch.equal(code, discretize_actions(multi, 64))
    # 不同序列大概率不同码
    assert len(set(code.tolist())) > 1


# ================================================================ 探针确定性
def test_single_agent_probe_is_deterministic():
    agent = single_agent()
    p1 = agent.nsp_probe()
    p2 = agent.nsp_probe()
    assert p1, "探针必须有读数"
    assert p1 == p2, "同一份权重两次探针必须逐位一致"
    for key, value in p1.items():
        assert math.isfinite(value), f"{key}={value} 非有限值"
    for key in ("nsp_total", "nsp_infonce_acc", "ema_drift", "latent_norm"):
        assert key in p1, f"探针缺 {key}"


def test_marl_probe_is_deterministic_and_prefixed():
    meta = marl_agent()
    p1 = meta.nsp_probe()
    p2 = meta.nsp_probe()
    assert p1 == p2, "MARL 探针必须可复现"
    for role in ("intent", "subintent", "slot"):
        assert f"{role}_nsp_total" in p1, f"缺 {role} 专家的 NSP 读数"
    assert meta.nsp_metrics_snapshot(), "训练滚动指标也要能读出来"


def test_gate_records_nsp_metrics_without_new_baseline_fields():
    from regression_gate import GateMetrics, GateThresholds

    # 旧基线里没有 nsp 字段 → 必须容错，否则 gate 直接报基础设施错误
    old = GateMetrics.from_dict({"overall_accuracy": 0.8, "macro_accuracy": 0.7,
                                 "sample_count": 100, "per_intent": {},
                                 "set_hash": "x"})
    assert old.nsp == {}
    assert old._nsp_values("nsp_total") == []

    thr = GateThresholds.from_dict({"tolerance": 0.05})
    assert thr.nsp_max_total is None and thr.nsp_min_infonce_acc is None, \
        "旧基线的阈值必须解析出 None（只记录不判定）"
    assert "nsp_max_total" in thr.to_dict()

    cur = GateMetrics.from_dict(old.to_dict())
    cur.nsp = {"nsp_total": 3.0, "intent_nsp_total": 5.0}
    assert cur._nsp_values("nsp_total") == [3.0, 5.0]
    assert max(cur._nsp_values("nsp_total")) == 5.0


# ================================================================ 消融 & 兼容
def test_disable_stops_gradients_and_imagine():
    nsp = _fresh_nsp()
    nsp.disable()
    assert nsp.trainable_parameters() == []
    nsp.update_ema()          # 不应抛异常，也不应改变目标
    assert nsp.ema_drift() == 0.0

    b = 4
    out, _ = nsp.imagine_q_target(torch.randn(b, 268),
                                  torch.randint(0, 11, (b,)), torch.randn(b),
                                  alpha=0.5)
    assert out is not None
    nsp.enable()
    assert len(nsp.trainable_parameters()) > 0


def test_q_network_state_dict_has_no_new_nsp_keys():
    """NSP 必须是旁挂模块：q_network 结构不变，旧 checkpoint 才能 strict 加载。

    （旧版 ``NSPHead`` 仍留在 q_network 里，删掉会让已有 checkpoint 加载失败；
    新增的 transition / projector / bridge / target 一律在 ``nsp`` key 下。）
    """
    from transformer_arch import TransformerDuelingQNetworkWithNSP

    net = TransformerDuelingQNetworkWithNSP(state_dim=268, d_model=128,
                                            num_actions=11)
    keys = set(net.state_dict().keys())
    new_parts = ("transition.", "projector.", "bridge.", "target.")
    assert not any(k.startswith(new_parts) for k in keys), sorted(keys)

    # 反序列化权重（结构一致）必须成功
    net2 = TransformerDuelingQNetworkWithNSP(state_dim=268, d_model=128,
                                             num_actions=11)
    net2.load_state_dict(net.state_dict())

    # 宿主 AgentRL 侧：新增权重只挂在 self.nsp 下
    agent = single_agent()
    nsp_keys = set(agent.nsp.state_dict().keys())
    assert nsp_keys and not (nsp_keys & keys), \
        "NSP 权重不得和 q_network 权重重名"


def test_train_step_runs_with_and_without_nsp():
    agent = single_agent()
    d = 268
    for k in range(200):
        s = np.random.randn(d).astype(np.float32)
        agent.replay_buffer.push(s, k % 11, 1.0, np.random.randn(d).astype(np.float32),
                                 done=(k % 7 == 6))

    agent.train_step()
    assert agent.nsp.metrics.count > 0, "开启时应当记录 NSP 滚动指标"
    snap = agent.nsp_metrics_snapshot()
    assert "nsp_total" in snap and math.isfinite(snap["nsp_total"])

    agent.nsp.disable()
    before = agent.training_step
    agent.train_step()
    assert agent.training_step == before + 1, "消融后 DQN 仍在训练"
    assert agent._imagine_alpha() == 0.0
    agent.nsp.enable()


def test_marl_imagine_alpha_ramp_and_segment_split():
    """MARL 侧的 α ramp（warmup 内恒 0）与 H 步轨迹段切分。"""
    from marl_agent import MetaAgent

    meta = MetaAgent.__new__(MetaAgent)
    meta.nsp_imagine_alpha = 0.2
    meta.nsp_imagine_warmup = 50
    meta.training_step = 10
    assert abs(meta._imagine_alpha() - 0.04) < 1e-9, "warmup 中段应线性爬升"
    meta.training_step = 50
    assert abs(meta._imagine_alpha() - 0.2) < 1e-9, "warmup 之后应到配置上限"

    done = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    assert MetaAgent._nsp_segments(done, 1) == [0, 3], \
        "2 步段可用、3 步段（含末位 done）不可用"
    assert MetaAgent._nsp_segments(torch.ones(6), 2) == [], \
        "全 done 的单步 episode 不该产出多步片段"


def test_checkpoint_roundtrip_keeps_nsp_state(tmp_path=None):
    agent = single_agent()
    path = os.path.join(TMP_ROOT, "nsp_roundtrip.pth")
    agent.save(path, save_replay=False)
    import torch as _torch
    ckpt = _torch.load(path, map_location="cpu", weights_only=False)
    assert "nsp" in ckpt, "save 必须写入 NSP 权重"
    assert "nsp_enabled" in ckpt

    from agent_rl import AgentRL
    restored = AgentRL(model_path=path, marl_vote=False)
    assert restored.training_step == agent.training_step
    for (k1, v1), (k2, v2) in zip(agent.nsp.state_dict().items(),
                                  restored.nsp.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2), f"{k1} 未恢复"


def test_marl_checkpoint_roundtrip_keeps_nsp_state():
    meta = marl_agent()
    path = os.path.join(TMP_ROOT, "marl_nsp_roundtrip.pth")
    meta.save(path)
    import torch as _torch
    ckpt = _torch.load(path, map_location="cpu", weights_only=False)
    for key in ("intent_nsp", "subintent_nsp", "slot_nsp", "nsp_enabled"):
        assert key in ckpt, f"MARL checkpoint 缺 {key}"

    from marl_agent import MetaAgent
    restored = MetaAgent(model_path=path)
    for role in ("intent_agent", "subintent_agent", "slot_agent"):
        a, b = getattr(meta, role), getattr(restored, role)
        assert a.nsp.enabled == b.nsp.enabled
        for (k1, v1), (k2, v2) in zip(a.nsp.state_dict().items(),
                                      b.nsp.state_dict().items()):
            assert k1 == k2 and torch.allclose(v1, v2), f"{role}.{k1} 未恢复"


def test_old_marl_checkpoint_without_nsp_still_loads(tmp_path=None):
    """旧 checkpoint 没有 nsp key → 必须容错，并让目标副本对齐在线编码器。"""
    meta = marl_agent()
    path = os.path.join(TMP_ROOT, "marl_old.pth")
    meta.save(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("intent_nsp", "subintent_nsp", "slot_nsp", "nsp_enabled"):
        ckpt.pop(key, None)
    torch.save(ckpt, path)

    from marl_agent import MetaAgent
    restored = MetaAgent(model_path=path)
    assert restored.loaded_from_disk
    for role in ("intent_agent", "subintent_agent", "slot_agent"):
        agent = getattr(restored, role)
        assert agent.nsp.ema_drift() < 1e-6, \
            f"{role}: 缺 nsp key 时目标副本必须 refresh 到在线编码器"


# ================================================================ 逐项消融
def test_apply_nsp_ablation_is_reversible_on_single_agent():
    """八个档位各自只动该动的字段，且复位后逐位回到构造默认值。"""
    from nsp_latent import ABLATION_SPECS, apply_nsp_ablation, nsp_enabled

    agent = single_agent()
    defaults = dict(agent.nsp._ablation_defaults)
    orig_spec = agent.nsp_ablate
    orig_alpha = agent.nsp_imagine_alpha
    try:
        cases = {
            "none": lambda: nsp_enabled(agent),
            "off": lambda: not nsp_enabled(agent),
            "no_ema": lambda: agent.nsp.ema_tau == 1.0,
            "no_infonce": lambda: agent.nsp.infonce_weight == 0.0,
            "no_bridge": lambda: agent.nsp.bridge_weight == 0.0,
            "no_hstep": lambda: agent.nsp.horizon == 1,
            "no_imagine": lambda: agent.nsp_imagine_alpha == 0.0,
            "deterministic": lambda: agent.nsp.deterministic,
        }
        assert set(cases) == set(ABLATION_SPECS), "档位表与用例表必须一一对应"
        for spec, check in cases.items():
            touched = apply_nsp_ablation(agent, spec)
            assert agent.nsp_ablate == spec, f"{spec} 未记到宿主上"
            assert check(), f"档位 {spec} 没有生效"
            if spec != "none":
                assert touched, f"档位 {spec} 不该是空改动"

        apply_nsp_ablation(agent, "none")
        assert nsp_enabled(agent) and not agent.nsp.deterministic
        for key, value in defaults.items():
            assert getattr(agent.nsp, key) == value, f"{key} 未复位"
        assert agent.nsp_imagine_alpha == orig_alpha, "想象权重未复位"
        assert apply_nsp_ablation(agent, "none") == [], "连续两次施加应幂等"
    finally:
        apply_nsp_ablation(agent, orig_spec)


def test_apply_nsp_ablation_covers_all_three_experts():
    from nsp_latent import apply_nsp_ablation, nsp_enabled, nsp_modules

    meta = marl_agent()
    orig_spec = meta.nsp_ablate
    orig_alpha = meta.nsp_imagine_alpha
    try:
        assert len(nsp_modules(meta)) == 3, "三专家各挂一份 NSP 实例"
        apply_nsp_ablation(meta, "no_hstep")
        for role, agent in meta._agents.items():
            assert agent.nsp.horizon == 1, f"{role} 专家没被改到"
        apply_nsp_ablation(meta, "off")
        assert not nsp_enabled(meta)
        assert all(not a.nsp.enabled for a in meta._agents.values())

        apply_nsp_ablation(meta, "none")
        expect = meta.intent_agent.nsp._ablation_defaults["horizon"]
        for agent in meta._agents.values():
            assert agent.nsp.enabled and agent.nsp.horizon == expect
        assert meta.nsp_imagine_alpha == meta.nsp_imagine_alpha_default
        assert meta.nsp_probe(), "复位后探针仍要能出数"
    finally:
        apply_nsp_ablation(meta, orig_spec)


def test_deterministic_ablation_drops_noise():
    """点估计档：KL 恒 0、NLL 退化成 MSE、转移不再采样。"""
    nsp = _fresh_nsp()
    mu = torch.full((4, 128), 3.0)
    logvar = torch.full((4, 128), 2.0)
    z_pos = torch.randn(4, 128)
    mse = torch.nn.functional.mse_loss(mu, z_pos)

    assert float(nsp._kl(mu, logvar)) > 0.0, "默认档必须是分布参数化"
    kl_before = float(nsp._kl(mu, logvar))
    assert not torch.allclose(nsp._nll(mu, logvar, z_pos), mse), \
        "默认档必须是高斯 NLL 而不是 MSE"

    nsp.deterministic = True
    assert float(nsp._kl(mu, logvar)) == 0.0, "点估计档 KL 必须恒 0"
    assert torch.allclose(nsp._nll(mu, logvar, z_pos), mse), \
        "点估计档 NLL 必须退化成 MSE"
    assert torch.equal(nsp._step_latent(mu, logvar), mu), "点估计档不得采样"
    assert nsp.rollout(torch.randn(5, 128), None), "开关不能让 rollout 崩"

    nsp.deterministic = False
    assert float(nsp._kl(mu, logvar)) == kl_before, "关掉档位必须复原"
    assert not torch.equal(nsp._step_latent(mu, logvar), mu), \
        "默认档必须真的采样，否则随机建模等于没做"


def test_ablation_spec_survives_checkpoint(tmp_path=None):
    """档位写进 checkpoint；旧 checkpoint 只有 nsp_enabled 时按 off 回退。"""
    from nsp_latent import apply_nsp_ablation, nsp_enabled

    agent = single_agent()
    path = os.path.join(TMP_ROOT, "nsp_ablate.pth")
    orig_spec = agent.nsp_ablate
    try:
        apply_nsp_ablation(agent, "no_hstep")
        agent.save(path, save_replay=False)

        from agent_rl import AgentRL
        restored = AgentRL(model_path=path, marl_vote=False)
        assert restored.nsp_ablate == "no_hstep", "档位没存进 checkpoint"
        assert restored.nsp.horizon == 1, "档位加载后没生效"

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt.pop("nsp_ablate", None)
        ckpt["nsp_enabled"] = False
        torch.save(ckpt, path)
        legacy = AgentRL(model_path=path, marl_vote=False)
        assert legacy.nsp_ablate == "off" and not nsp_enabled(legacy), \
            "只有 nsp_enabled 的旧 checkpoint 必须退回 off 档"
    finally:
        apply_nsp_ablation(agent, orig_spec)


def test_train_strong_parses_ablation_flags():
    """``--nsp-ablate no_hstep`` 的取值不能被当成 steps 触发 int() 崩溃。"""
    import train_strong as ts

    saved = sys.argv
    try:
        sys.argv = ["train_strong.py", "100", "50", "--nsp-ablate", "no_hstep"]
        assert ts._positional_args() == ["100", "50"]
        assert ts._flag_value("--nsp-ablate") == "no_hstep"

        sys.argv = ["train_strong.py", "--agent=marl", "--fresh"]
        assert ts._positional_args() == []
        assert ts._flag_value("--agent", "single") == "marl"

        sys.argv = ["train_strong.py", "--ablate-nsp", "20"]
        assert ts._positional_args() == ["20"]
        assert ts._flag_value("--nsp-ablate", None) is None
    finally:
        sys.argv = saved


# ================================================================ harness
TESTS = [
    test_ablation_spec_survives_checkpoint,
    test_apply_nsp_ablation_covers_all_three_experts,
    test_apply_nsp_ablation_is_reversible_on_single_agent,
    test_build_nsp_does_not_perturb_global_rng,
    test_checkpoint_roundtrip_keeps_nsp_state,
    test_disable_stops_gradients_and_imagine,
    test_deterministic_ablation_drops_noise,
    test_gate_records_nsp_metrics_without_new_baseline_fields,
    test_imagine_alpha_ramps_from_zero,
    test_imagine_q_target_alpha_gate_and_clamp,
    test_infonce_negatives_prevent_collapse,
    test_marl_checkpoint_roundtrip_keeps_nsp_state,
    test_marl_imagine_alpha_ramp_and_segment_split,
    test_marl_probe_is_deterministic_and_prefixed,
    test_old_marl_checkpoint_without_nsp_still_loads,
    test_online_encoder_is_not_registered_as_submodule,
    test_optimizer_has_no_duplicate_parameters,
    test_q_network_state_dict_has_no_new_nsp_keys,
    test_refresh_target_snaps_after_reinit,
    test_segment_losses_produces_finite_rollout_error,
    test_segments_do_not_cross_done_or_seg_boundary,
    test_single_agent_probe_is_deterministic,
    test_supervise_uses_nll_and_kl_not_pure_mse,
    test_target_is_frozen_and_tracks_online,
    test_train_step_runs_with_and_without_nsp,
    test_train_strong_parses_ablation_flags,
    test_transition_head_outputs_mu_and_logvar,
    test_discretize_actions_handles_multi_discrete,
]


def run_all() -> bool:
    print("=" * 70)
    print("  NSP 潜空间测试")
    print("=" * 70)

    discovered = {name for name, obj in globals().items()
                  if name.startswith("test_") and callable(obj)}
    registered = {case.__name__ for case in TESTS}
    if discovered != registered:
        print(f"[FAIL] TESTS 注册表与实际用例不一致："
              f"漏登记={sorted(discovered - registered)} "
              f"多余={sorted(registered - discovered)}")
        return False

    started = time.time()
    failures = []
    for case in TESTS:
        print(f"\n--- {case.__name__} ---")
        try:
            case()
            print("[PASS]")
        except AssertionError as err:
            print(f"[FAIL] {err}")
            failures.append(case.__name__)
        except Exception as err:  # noqa: BLE001  —— 单个用例炸掉不能拖垮整轮
            print(f"[ERROR] {type(err).__name__}: {err}")
            failures.append(f"{case.__name__} ({type(err).__name__})")

    print("\n" + "=" * 70)
    if failures:
        print(f"  失败 {len(failures)}/{len(TESTS)}: {', '.join(failures)}")
        print("=" * 70)
        return False
    print(f"  全部通过 {len(TESTS)}/{len(TESTS)}  "
          f"({time.time() - started:.1f}s)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
