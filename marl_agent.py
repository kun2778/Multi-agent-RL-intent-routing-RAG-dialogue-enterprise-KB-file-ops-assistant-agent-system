"""Multi-Agent RL (MARL) 意图路由委员会 —— 联盟训练 + 集成 Critic + MAPPO。

与单 Agent（agent_rl.AgentRL: Double DQN + PER + 蒸馏）**完全不同的训练方法**，
且在初始化、超参数、观测三个维度上刻意错开，为 6 路投票提供真正的策略多样性。

┌────────────┬──────────────────────────────┬────────────────────────────────────────┐
│ 维度       │ 单 Agent (agent_rl)          │ MARL (本模块)                          │
├────────────┼──────────────────────────────┼────────────────────────────────────────┤
│ 训练算法   │ Double DQN + PER + 蒸馏      │ MAPPO（PPO-Clip，on-policy）           │
│ 信用分配   │ 无（单一 TD 误差）           │ 边际贡献差分奖励 + 反事实基线(COMA)    │
│ 价值评估   │ 单一目标网络 + 软更新        │ 3 元集成 Critic（bagging + 悲观取 min）│
│ 协调机制   │ 无（固定权重平均投票）       │ MetaAgent 学习式源权重(EXP3) + 协调奖励│
│ 非平稳性   │ 无（静态环境 + 经验回放）    │ 联盟存档多代策略 + 对手上下文特征      │
│ 探索       │ ε-greedy (0.3→0.02,0.998)   │ 熵正则 + 熵感知 ε(0.30→0.03,0.996)     │
│ 初始化     │ 默认初始化，lr 全局 3e-4     │ 每角色独立种子 + 正交/Xavier/Kaiming    │
│ 超参数     │ γ 全局、无 clip/熵           │ 每角色独立 γ/λ/clip/熵/epochs/lr       │
│ 观测       │ [原始语义256 | 状态12] =268  │ [语义@正交基·掩码256 | 状态12 | 角色4]  │
│            │                              │                                     =272│
│ 策略集成   │ 无                           │ 当前策略 + 联盟 Top-K 加权 + 蒸馏      │
└────────────┴──────────────────────────────┴────────────────────────────────────────┘

设计要点与需求的对应：
  * 协调（cooperative）：意图路由是合作场景 —— 多样性必须配合协调机制才有增益。
    因此 MetaAgent 用 EXP3 在线学习各路源权重；意图 Agent 的奖励里含
    「与最终共识一致」的协调项与「去掉我这一票共识会翻转」的边际贡献项；
    Critic 是集中式的（本地观测 + 团队上下文 CTX），即 CTDE。
  * 信用分配：r_i = (0.6 + 0.4 * marginal_i) * R_team + coord_bonus，
    再叠加 COMA 式反事实优势 A_cf = Q(s,a) - Σ_a' π(a'|s) Q(s,a')
    （Q 取策略网络 logits，已与 GAE 优势各自归一化后加权融合）。
  * 非平稳性：AlphaStar 式联盟（存档 → 采样 → 晋升/淘汰），每个 rollout 周期
    采样一个对手世代，对手身份/强度编码进观测（对手建模），奖励里含 head-to-head
    胜负项（仅在团队判定正确时生效，避免把系统推向互相拆台）；
    另有 QueryOpponent 扰动 bandit 持续改变训练输入分布。
    on-policy 更新天然丢弃陈旧样本，避免 replay 放大非平稳性。
  * 探索与稳定学习：熵奖励（每角色不同系数）、策略采样探索、PPO-Clip、
    价值裁剪、梯度裁剪、KL 早停、优势归一化、奖励尺度归一化、
    集成 Critic 悲观(min)目标 + bagging 多样性。
  * 策略多样性：不同初始化（种子/方案）、不同超参数、不同观测（投影基/掩码/辅助通道/维度）。
  * 集成 Critic：K=3 价值网络 bagging，目标取 min 降低过估计，分歧度作为认知不确定性
    反馈进观测与投票置信度校准。
  * 集成策略：推断时当前策略与联盟成员按胜率加权平均（软投票），
    成员间 JS 散度 → 校准置信度（不确定性估计），并周期性蒸馏回单个网络。
  * 竞争/混合动机：head-to-head 胜负信号让多样性不易被单一对手模式利用。
"""

import copy
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent_state import (
    IntentType, QuerySubIntent, SlotType, AgentState,
    SHORT_LABELS, QUERY_SUBINTENT_LABELS
)
from query_rewriter import QueryRewriter
from llm_intent import LLMIntentClassifier
from intent_retriever import get_intent_retriever
from transformer_arch import (
    TransformerIntentQNetwork, TransformerSubIntentQNetwork,
    TransformerSlotExtractionNetwork, EngramConditionalMemory,
)
from agent_rl import AgentRL, round_reward, should_clarify

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MARL_MODEL_PATH = os.path.join(MODEL_DIR, "marl_agents.pth")

# 检查点格式：训练方法由 DQN 改为 MAPPO、观测由 268 改为 272，
# 旧检查点权重不兼容（加载时忽略并重新初始化）。
MARL_CKPT_FORMAT = 2

SEM_DIM = 256
STATE_DIM = 12
AUX_DIM = 4
OBS_DIM = SEM_DIM + STATE_DIM + AUX_DIM   # 272 ≠ 单 Agent 的 268（观测不同）
CTX_DIM = 16                              # 集中式 Critic 的团队上下文
ENSEMBLE_K = 3                            # 集成 Critic 成员数
MAX_LEAGUE = 3                            # 联盟存档上限
LEAGUE_TOP_K = 2                          # 推断时参与集成的联盟成员数
ROLLOUT_SIZE = 64                         # 攒够这么多 transition 触发一次 MAPPO 更新
MIN_STEPS_PER_AGENT = 12                  # 单个 Agent 参与更新的最低样本数
MAX_STEPS_BEFORE_FLUSH = 20               # 在线学习最多等这么多轮就强制更新
LEAGUE_INTERVAL = 4                       # 每多少次 PPO 更新做一次联盟评估/晋升
PROMOTE_THRESHOLD = 0.55                  # 对当前策略的 head-to-head 胜率阈值
EXP3_LR = 0.08                            # 源权重协调器（EXP3/Hedge）学习率
EXP3_FLOOR = 0.25                         # 源权重下限，防止单路被饿死
COORD_BONUS = 0.35                        # 与最终共识一致的协调奖励
H2H_BONUS = 0.30                          # 联盟 head-to-head 胜负奖励
OPP_PROB = 0.35                           # 训练时施加查询扰动（对手生成）的概率


# ================================================================
#  0. 基础工具
# ================================================================
def _init_network(module: nn.Module, scheme: str, gain: float, seed: int):
    """按角色指定的方案/种子初始化权重（不同初始化 → 策略多样性）。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for m in module.modules():
            if isinstance(m, nn.Linear):
                if scheme == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                else:  # kaiming
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def _normalize(t: torch.Tensor) -> torch.Tensor:
    """优势归一化（均值 0 / 方差 1），常数向量安全返回 0。"""
    return (t - t.mean()) / t.std(unbiased=False).clamp_min(1e-6)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-12)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS 散度（自然对数），∈[0, ln2]。"""
    p = np.clip(p, 1e-9, 1.0)
    q = np.clip(q, 1e-9, 1.0)
    m = 0.5 * (p + q)
    kl_pm = float((p * (np.log(p) - np.log(m))).sum())
    kl_qm = float((q * (np.log(q) - np.log(m))).sum())
    return 0.5 * (kl_pm + kl_qm)


class RunningScale:
    """奖励尺度归一化（只缩放不减均值，保留符号），稳定 critic 与优势估计。"""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        self.count += 1
        d = x - self.mean
        self.mean += d / self.count
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(math.sqrt(self.m2 / (self.count - 1)), 1e-3)

    def scale(self, x: float) -> float:
        return float(np.clip(x / self.std, -10.0, 10.0))

    def state_dict(self) -> Dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, d: Dict[str, float]):
        self.count = int(d.get("count", 0))
        self.mean = float(d.get("mean", 0.0))
        self.m2 = float(d.get("m2", 0.0))


# ================================================================
#  1. 每角色超参数（与单 Agent 显著不同，且角色之间互不相同）
# ================================================================
@dataclass(frozen=True)
class AgentHParams:
    name: str
    seed: int
    init_scheme: str
    init_gain: float
    obs_seed: int
    mask_rate: float
    lr: float
    critic_lr: float
    gamma: float
    gae_lambda: float
    clip: float
    entropy_coef: float
    value_coef: float
    cf_weight: float          # 反事实优势在最终优势中的占比
    ppo_epochs: int
    minibatch: int
    max_grad_norm: float
    kl_target: float


INTENT_HP = AgentHParams(
    name="intent", seed=1001, init_scheme="orthogonal", init_gain=0.02,
    obs_seed=1001, mask_rate=0.00,
    lr=1.5e-4, critic_lr=3.0e-4, gamma=0.99, gae_lambda=0.95, clip=0.20,
    entropy_coef=0.015, value_coef=0.5, cf_weight=0.35,
    ppo_epochs=4, minibatch=32, max_grad_norm=0.5, kl_target=0.012,
)

SUBINTENT_HP = AgentHParams(
    name="subintent", seed=2002, init_scheme="xavier", init_gain=1.0,
    obs_seed=2002, mask_rate=0.20,
    lr=3.0e-4, critic_lr=5.0e-4, gamma=0.97, gae_lambda=0.90, clip=0.15,
    entropy_coef=0.030, value_coef=0.4, cf_weight=0.30,
    ppo_epochs=3, minibatch=32, max_grad_norm=0.6, kl_target=0.020,
)

SLOT_HP = AgentHParams(
    name="slot", seed=3003, init_scheme="kaiming", init_gain=1.0,
    obs_seed=3003, mask_rate=0.35,
    lr=1.0e-4, critic_lr=2.0e-4, gamma=0.95, gae_lambda=0.85, clip=0.25,
    entropy_coef=0.010, value_coef=0.6, cf_weight=0.25,
    ppo_epochs=2, minibatch=16, max_grad_norm=0.4, kl_target=0.010,
)


# ================================================================
#  2. 角色专属观测编码（不同观测 → 策略多样性）
# ================================================================
class ObservationEncoder:
    """把 (语义向量, 状态向量, 角色辅助通道) 编码为 OBS_DIM 维策略观测。

    与单 Agent 的 [原始语义 | 状态] = 268 维不同：
      * 语义块先乘上该角色**独立种子的正交基**（不同表征坐标系）；
      * 语义块按 mask_rate 随机丢弃部分维度（不同观测子集）；
      * 尾部追加 4 维角色专属辅助通道（对手上下文 / 检索 / 规则 / 正则特征）。
    """

    def __init__(self, seed: int, mask_rate: float):
        rng = np.random.RandomState(seed)
        basis, _ = np.linalg.qr(rng.randn(SEM_DIM, SEM_DIM))
        self.basis = basis.astype(np.float32)
        self.keep = (rng.rand(SEM_DIM) >= mask_rate).astype(np.float32)

    def encode(self, sem: np.ndarray, state_vec: np.ndarray, aux: np.ndarray) -> np.ndarray:
        v = (sem.astype(np.float32) @ self.basis) * self.keep
        norm = float(np.linalg.norm(v))
        if norm > 1e-8:
            v = v / norm
        return np.concatenate([v, state_vec.astype(np.float32),
                               aux.astype(np.float32)]).astype(np.float32)


# ================================================================
#  3. 集成 Critic（bagging + 悲观 min，降低过估计）
# ================================================================
class CriticEnsemble(nn.Module):
    """K 个价值网络：训练时各自 bagging 采样，读数时取 min（悲观）与 std（不确定性）。"""

    def __init__(self, obs_dim: int, k: int = ENSEMBLE_K, hidden: int = 128, seed: int = 7):
        super().__init__()
        self.members = nn.ModuleList()
        for i in range(k):
            head = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            _init_network(head, "orthogonal", gain=0.5, seed=seed + i * 97)
            # 乐观初始化：价值起点偏高，鼓励早期探索
            with torch.no_grad():
                head[-1].bias.fill_(0.3 + 0.2 * i)
            self.members.append(head)

    def forward(self, x: torch.Tensor, reduction: str = "min") -> Tuple[torch.Tensor, torch.Tensor]:
        vals = torch.stack([m(x).squeeze(-1) for m in self.members], dim=-1)  # (B, K)
        v = vals.mean(dim=-1) if reduction == "mean" else vals.min(dim=-1).values
        std = vals.std(dim=-1, unbiased=False)
        return v, std


# ================================================================
#  4. 联盟（AlphaStar 式存档-采样-晋升）
# ================================================================
class LeaguePlayer:
    """联盟中的一个历史策略世代（快照）。"""

    def __init__(self, pid: int, net: nn.Module, win_rate: float,
                 step: int, hp_name: str):
        self.pid = pid
        self.net = net
        self.net.eval()
        self.win_rate = win_rate
        self.step = step
        self.hp_name = hp_name


class League:
    """策略联盟：多样对手来源，防止单一策略过拟合 + 提供非平稳性。"""

    def __init__(self, factory: Callable[[], nn.Module], max_size: int = MAX_LEAGUE):
        self.factory = factory
        self.max_size = max_size
        self.players: List[LeaguePlayer] = []
        self._pid = 1

    def __len__(self):
        return len(self.players)

    def sample(self) -> Optional[LeaguePlayer]:
        """按胜率加权采样对手，25% 概率均匀采样（保证多样性、避免锁定单一强手）。"""
        if not self.players:
            return None
        if random.random() < 0.25:
            return random.choice(self.players)
        weights = [max(p.win_rate, 0.05) for p in self.players]
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for p, w in zip(self.players, weights):
            acc += w
            if r <= acc:
                return p
        return self.players[-1]

    def top_players(self, k: int = LEAGUE_TOP_K) -> List[LeaguePlayer]:
        return sorted(self.players, key=lambda p: -p.win_rate)[:k]

    def promote(self, net: nn.Module, win_rate: float, step: int,
                hp_name: str) -> Optional[LeaguePlayer]:
        """晋升当前策略：满员时淘汰胜率最低者（保留历史多样性）。"""
        if len(self.players) >= self.max_size:
            weakest = min(self.players, key=lambda p: p.win_rate)
            self.players.remove(weakest)
        player = LeaguePlayer(self._pid, copy.deepcopy(net), win_rate, step, hp_name)
        player.net.eval()
        self._pid += 1
        self.players.append(player)
        return player

    @torch.no_grad()
    def aggregate_probs(self, obs: np.ndarray, k: int = LEAGUE_TOP_K) -> Optional[np.ndarray]:
        """联盟集成策略：Top-K 成员按胜率加权的 softmax 平均。"""
        members = self.top_players(k)
        if not members:
            return None
        x = torch.FloatTensor(obs).unsqueeze(0)
        acc = None
        wsum = 0.0
        for p in members:
            logits = p.net(x).squeeze(0).numpy()
            w = max(p.win_rate, 0.05)
            part = _softmax_np(logits) * w
            acc = part if acc is None else acc + part
            wsum += w
        if acc is None or wsum <= 0:
            return None
        return acc / wsum

    def state_dict(self) -> Dict[str, Any]:
        return {
            "pid": self._pid,
            "players": [
                {"pid": p.pid, "net": p.net.state_dict(), "win_rate": p.win_rate,
                 "step": p.step, "hp_name": p.hp_name}
                for p in self.players
            ],
        }

    def load_state_dict(self, d: Dict[str, Any]):
        self.players = []
        self._pid = int(d.get("pid", 1))
        for item in d.get("players", []):
            net = self.factory()
            try:
                net.load_state_dict(item["net"])
            except Exception:
                continue
            self.players.append(LeaguePlayer(
                int(item.get("pid", 0)), net,
                float(item.get("win_rate", 0.5)),
                int(item.get("step", 0)),
                str(item.get("hp_name", "intent")),
            ))


# ================================================================
#  5. 对手查询生成器（意图保持的扰动 bandit → 训练分布非平稳 + 防过拟合）
# ================================================================
class QueryOpponent:
    """bandit 选择「如何改写用户查询」，使训练输入分布持续变化。

    所有扰动都**保持意图标签不变**（只加/减语气词、空白、切分），
    避免制造错误监督信号。
    """

    ARMS = ("identity", "polite_prefix", "casual_suffix",
            "strip_filler", "space_noise", "segment_split")

    def __init__(self, seed: int = 7):
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.counts = {a: 1 for a in self.ARMS}
        self.reward_sum = {a: 0.0 for a in self.ARMS}
        self.prefixes = ("请", "帮我", "麻烦", "你好，", "请问")
        self.suffixes = ("一下", "吧", "呢", "谢谢", "麻烦你了")
        self.fillers = ("请", "麻烦", "你", "一下", "吧", "呢", "啊", "谢谢", "你好")

    @property
    def weights(self) -> np.ndarray:
        scores = np.array([self.reward_sum[a] / self.counts[a] for a in self.ARMS])
        scores = scores / (np.abs(scores).max() + 1e-6)
        w = np.exp(scores / 0.5)
        w = 0.75 * w / w.sum() + 0.25 / len(self.ARMS)   # 保底探索
        return w

    def choose(self) -> str:
        w = self.weights
        idx = int(self.rng.choice(len(self.ARMS), p=w))
        return self.ARMS[idx]

    def apply(self, query: str, arm: str) -> str:
        q = query.strip()
        if not q or arm == "identity":
            return query
        try:
            if arm == "polite_prefix":
                return f"{self.prefixes[self.rng.randint(len(self.prefixes))]}{q}"
            if arm == "casual_suffix":
                return f"{q}{self.suffixes[self.rng.randint(len(self.suffixes))]}"
            if arm == "strip_filler":
                out = q
                for f in self.fillers:
                    if len(out) > 4 and out.startswith(f):
                        out = out[len(f):]
                return out or q
            if arm == "space_noise":
                if len(q) < 3:
                    return q
                pos = int(self.rng.randint(1, len(q)))
                return q[:pos] + " " + q[pos:]
            if arm == "segment_split":
                return " ".join(list(q)) if len(q) <= 12 else q
        except Exception:
            return query
        return query

    def update(self, arm: str, reward: float):
        self.counts[arm] = self.counts.get(arm, 1) + 1
        self.reward_sum[arm] = self.reward_sum.get(arm, 0.0) + reward

    def state_dict(self) -> Dict[str, Any]:
        return {"counts": dict(self.counts), "reward_sum": dict(self.reward_sum)}

    def load_state_dict(self, d: Dict[str, Any]):
        self.counts.update({k: int(v) for k, v in (d.get("counts") or {}).items()})
        self.reward_sum.update({k: float(v) for k, v in (d.get("reward_sum") or {}).items()})


# ================================================================
#  6. 过渡样本
# ================================================================
@dataclass
class Transition:
    agent: str
    obs: np.ndarray
    ctx: np.ndarray
    action: Any
    logprob: float
    reward: float
    done: bool
    value: float
    next_obs: np.ndarray
    next_ctx: np.ndarray
    opp_pred: int = -1


# ================================================================
#  7. 个体策略（PPO-Clip + 反事实优势 + 集成 Critic）
# ================================================================
class PolicyAgent:
    """MARL 中的个体策略基类：角色专属观测 + 独立初始化/超参 + MAPPO 更新。"""

    hp: AgentHParams

    def __init__(self, hp: AgentHParams, encoder: ObservationEncoder):
        self.hp = hp
        self.encoder = encoder
        self.critic = CriticEnsemble(OBS_DIM + CTX_DIM, ENSEMBLE_K, seed=hp.seed + 13)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hp.critic_lr, eps=1e-5)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=hp.lr, eps=1e-5)

    # ---- 观测 ----
    def encode(self, sem: np.ndarray, state_vec: np.ndarray, aux: np.ndarray) -> np.ndarray:
        return self.encoder.encode(sem, state_vec, aux)

    # ---- 前向 ----
    def policy_outputs(self, obs_t: torch.Tensor) -> Any:
        return self.q_network(obs_t)

    @torch.no_grad()
    def probs_np(self, obs: np.ndarray) -> np.ndarray:
        self.q_network.eval()
        x = torch.FloatTensor(obs).unsqueeze(0)
        probs = self._probs_from_output(self.policy_outputs(x)).squeeze(0).numpy()
        self.q_network.train()
        return probs

    def _probs_from_output(self, out: Any) -> torch.Tensor:
        return F.softmax(out, dim=-1)

    # ---- 采样 ----
    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool) -> Tuple[Any, float, float, np.ndarray]:
        """返回 (action, logprob, entropy, probs)。explore=True 时按当前策略采样。"""
        self.q_network.eval()
        x = torch.FloatTensor(obs).unsqueeze(0)
        out = self.policy_outputs(x)
        probs = self._probs_from_output(out).squeeze(0)
        action, logprob, entropy = self._sample(probs, explore)
        self.q_network.train()
        return action, float(logprob), float(entropy), probs.numpy()

    def _sample(self, probs: torch.Tensor, explore: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample() if explore else torch.as_tensor(
            int(probs.argmax()), dtype=torch.int64)
        logp = dist.log_prob(action)
        ent = dist.entropy()
        return action, logp, ent

    # ---- 批量评估 ----
    def _logp_entropy(self, out: Any, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self._probs_from_output(out)
        logp = torch.log(probs.gather(1, actions.view(-1, 1)).clamp_min(1e-9)).squeeze(1)
        ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)
        return logp, ent

    def stack_actions(self, trs: List[Transition]) -> torch.Tensor:
        return torch.as_tensor([t.action for t in trs], dtype=torch.int64)

    def counterfactual_advantage(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """COMA 反事实基线：A = Q(s,a) - Σ_a' π(a'|s) Q(s,a')，Q 取网络 logits。"""
        with torch.no_grad():
            out = self.policy_outputs(obs)
            logits = out if torch.is_tensor(out) else out
            probs = self._probs_from_output(logits)
            q_a = logits.gather(1, actions.view(-1, 1)).squeeze(1)
            baseline = (probs * logits).sum(-1)
        return q_a - baseline

    # ---- 子类必须实现 ----
    def build_aux(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def extra_logp_entropy(self, *args, **kwargs):
        """多离散（Slot）子类覆盖。"""
        return None


class IntentPolicy(PolicyAgent):
    """主意图 Agent：11 类意图，观测量 = 272 维（含联盟对手上下文）。"""

    INTENT_LIST = [
        IntentType.ADD_FILE, IntentType.DELETE_FILE, IntentType.MODIFY_FILE,
        IntentType.REFACTOR, IntentType.QUERY, IntentType.DEEP_SEARCH, IntentType.VAGUE,
        IntentType.FILE_ANALYSIS, IntentType.FILE_SCAN, IntentType.REPORT_GENERATE,
        IntentType.FILENAME_QUERY,
    ]

    def __init__(self, hp: AgentHParams = INTENT_HP):
        self.q_network = TransformerIntentQNetwork(
            OBS_DIM, 128, len(self.INTENT_LIST), num_loops=4)
        self.action_map = {i: it for i, it in enumerate(self.INTENT_LIST)}
        self.reverse_map = {it: i for i, it in self.action_map.items()}
        super().__init__(hp, ObservationEncoder(hp.obs_seed, hp.mask_rate))
        _init_network(self.q_network, hp.init_scheme, hp.init_gain, hp.seed)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=hp.lr, eps=1e-5)

        # 规则兜底（与单 Agent 的关键词规则同源，但权重由 MetaAgent 学习）
        self.rule_keywords = {
            IntentType.ADD_FILE: ["添加", "新增", "创建", "新建", "写入", "保存", "生成", "add", "create"],
            IntentType.DELETE_FILE: ["删除", "移除", "清理", "删掉", "remove", "delete"],
            IntentType.MODIFY_FILE: ["修改", "更新", "编辑", "改成", "modify", "edit", "update"],
            IntentType.REFACTOR: ["整理", "重构", "归类", "归档", "重命名", "refactor", "organize"],
            IntentType.QUERY: ["查", "找", "搜索", "查询", "检索", "怎么", "什么", "search", "find",
                                "识别", "截屏", "截图", "ocr", "文字", "输出", "提取", "读取", "查看"],
            IntentType.DEEP_SEARCH: ["对比", "比较", "分析", "差异", "综合", "总结", "analyze"],
            IntentType.FILE_ANALYSIS: ["分析", "审查", "检查", "诊断", "评估", "只读分析"],
            IntentType.FILE_SCAN: ["扫描", "遍历", "列举", "统计", "文件扫描"],
            IntentType.REPORT_GENERATE: ["报告", "生成报告", "输出报告", "汇总"],
            IntentType.FILENAME_QUERY: ["最近", "最新", "文件名搜索", "搜索文件名",
                                         "查找文件", "recent", "latest", "newly"],
        }

    def rule_predict(self, query: str) -> Optional[IntentType]:
        q = query.lower()
        scores: Dict[IntentType, int] = {}
        for intent, keywords in self.rule_keywords.items():
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in q:
                    idx = q.find(kw_lower)
                    if idx == 0:
                        score += 2
                    elif q[idx - 1] in ' ,;:!?。，；：！？':
                        score += 2
                    else:
                        score += 1
            if score > 0:
                scores[intent] = score
        if not scores:
            return None
        return max(scores, key=scores.get)


class SubIntentPolicy(PolicyAgent):
    """子意图 Agent：QUERY 下的 12 类细分，观测掩码 20% + 检索/规则辅助通道。"""

    SUBINTENT_LIST = [
        QuerySubIntent.QUERY_METADATA, QuerySubIntent.QUERY_CONTENT,
        QuerySubIntent.QUERY_OCR, QuerySubIntent.QUERY_SUMMARY,
        QuerySubIntent.QUERY_TRANSLATE, QuerySubIntent.QUERY_EXTRACT,
        QuerySubIntent.QUERY_SEARCH, QuerySubIntent.QUERY_SYSTEM,
        QuerySubIntent.QUERY_OTHER,
        QuerySubIntent.QUERY_FILE_SCAN, QuerySubIntent.QUERY_FILE_ANALYZE,
        QuerySubIntent.QUERY_READONLY_REPORT,
    ]

    def __init__(self, hp: AgentHParams = SUBINTENT_HP):
        self.q_network = TransformerSubIntentQNetwork(
            OBS_DIM, 128, len(self.SUBINTENT_LIST), num_loops=4)
        self.action_map = {i: s for i, s in enumerate(self.SUBINTENT_LIST)}
        self.reverse_map = {s: i for i, s in self.action_map.items()}
        super().__init__(hp, ObservationEncoder(hp.obs_seed, hp.mask_rate))
        _init_network(self.q_network, hp.init_scheme, hp.init_gain, hp.seed)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=hp.lr, eps=1e-5)

        self.priority_keywords = {
            QuerySubIntent.QUERY_OCR: ["识别", "ocr", "文字识别", "截屏", "截图"],
            QuerySubIntent.QUERY_TRANSLATE: ["翻译", "translate", "译成"],
            QuerySubIntent.QUERY_SUMMARY: ["总结", "摘要", "概括", "简述"],
            QuerySubIntent.QUERY_EXTRACT: ["提取", "抽出", "抽取"],
            QuerySubIntent.QUERY_METADATA: ["信息", "大小", "修改时间"],
            QuerySubIntent.QUERY_CONTENT: ["内容", "读取", "查看内容"],
            QuerySubIntent.QUERY_SEARCH: ["搜索", "查找", "检索"],
            QuerySubIntent.QUERY_SYSTEM: ["系统", "程序", "软件"],
            QuerySubIntent.QUERY_FILE_SCAN: ["扫描", "遍历", "列举"],
            QuerySubIntent.QUERY_FILE_ANALYZE: ["分析", "审查", "检查"],
            QuerySubIntent.QUERY_READONLY_REPORT: ["报告", "审查报告", "生成报告"],
        }

    def rule_predict(self, query: str) -> Optional[QuerySubIntent]:
        q = query.lower()
        for sub_intent, keywords in self.priority_keywords.items():
            for kw in keywords:
                if kw.lower() in q:
                    return sub_intent
        return None


class SlotPolicy(PolicyAgent):
    """槽位提取 Agent：5 个离散头（multi-discrete），观测掩码 35% + 正则辅助通道。"""

    LANGUAGES = ["英文", "中文", "日文", "韩文", "法文", "德文", "其他"]
    TIME_RANGES = ["recent", "today", "yesterday", "this_week", "this_month", "other"]
    FORMATS = ["json", "csv", "markdown", "table", "other"]

    def __init__(self, hp: AgentHParams = SLOT_HP):
        self.q_network = TransformerSlotExtractionNetwork(OBS_DIM, 128, num_loops=4)
        super().__init__(hp, ObservationEncoder(hp.obs_seed, hp.mask_rate))
        _init_network(self.q_network, hp.init_scheme, hp.init_gain, hp.seed)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=hp.lr, eps=1e-5)

        import re
        self.file_pattern = re.compile(
            r"[A-Za-z]:\\[^\s]*?\.[A-Za-z0-9]{1,6}|[\w.\-\\/]+\.[A-Za-z0-9]{1,6}",
            re.IGNORECASE,
        )
        self.image_pattern = re.compile(
            r"[\w.\-\\/]+\.(png|jpg|jpeg|bmp|tiff|gif)", re.IGNORECASE)

    # ---- multi-discrete 处理 ----
    def _probs_from_output(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        # (B, 5 heads 拼接) —— 仅用于 act/entropy 的统一入口，head 尺寸不同故不拼接
        raise NotImplementedError("SlotPolicy 直接使用 head 字典，不走统一入口")

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool) -> Tuple[Any, float, float, np.ndarray]:
        self.q_network.eval()
        x = torch.FloatTensor(obs).unsqueeze(0)
        outputs = self.q_network(x)
        actions, logps, ents = [], 0.0, 0.0
        merged = []
        for name in ("file", "image", "language", "time", "format"):
            probs = outputs[name].squeeze(0)
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample() if explore else torch.as_tensor(int(probs.argmax()))
            actions.append(int(a))
            logps = logps + float(dist.log_prob(a))
            ents = ents + float(dist.entropy())
            merged.extend(probs.tolist())
        self.q_network.train()
        return tuple(actions), float(logps), float(ents), np.asarray(merged, dtype=np.float32)

    @torch.no_grad()
    def probs_np(self, obs: np.ndarray) -> np.ndarray:
        self.q_network.eval()
        x = torch.FloatTensor(obs).unsqueeze(0)
        outputs = self.q_network(x)
        merged = np.concatenate([outputs[n].squeeze(0).numpy()
                                 for n in ("file", "image", "language", "time", "format")])
        self.q_network.train()
        return merged

    def _logp_entropy(self, out: Dict[str, torch.Tensor],
                      actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logp = torch.zeros(actions.shape[0])
        ent = torch.zeros(actions.shape[0])
        for i, name in enumerate(("file", "image", "language", "time", "format")):
            probs = out[name].clamp_min(1e-9)
            logp = logp + torch.log(probs.gather(1, actions[:, i:i + 1])).squeeze(1)
            ent = ent + -(probs * torch.log(probs)).sum(-1)
        return logp, ent

    def stack_actions(self, trs: List[Transition]) -> torch.Tensor:
        return torch.as_tensor([list(t.action) for t in trs], dtype=torch.int64)

    @torch.no_grad()
    def counterfactual_advantage(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """多离散情形：用 logπ 作为 Q 的代理，A = logπ(a) - Σπ logπ = logπ(a) + H(π)。"""
        out = self.policy_outputs(obs)
        logp, ent = self._logp_entropy(out, actions)
        return logp + ent

    def predict(self, features: np.ndarray) -> Dict[str, Tuple[str, float]]:
        """兼容旧接口：返回各槽位 (取值, 置信度)。"""
        self.q_network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            outputs = self.q_network(x)
            results = {}
            fp = outputs["file"][0].numpy()
            results["file_path"] = ("present" if fp.argmax() else "absent", float(fp.max()))
            ip = outputs["image"][0].numpy()
            results["image_path"] = ("present" if ip.argmax() else "absent", float(ip.max()))
            lp = outputs["language"][0].numpy()
            results["language"] = (self.LANGUAGES[int(lp.argmax())], float(lp.max()))
            tp = outputs["time"][0].numpy()
            results["time_range"] = (self.TIME_RANGES[int(tp.argmax())], float(tp.max()))
            fp2 = outputs["format"][0].numpy()
            results["format"] = (self.FORMATS[int(fp2.argmax())], float(fp2.max()))
        self.q_network.train()
        return results

    def rule_extract(self, query: str) -> Dict[str, Optional[str]]:
        slots: Dict[str, Optional[str]] = {}
        file_matches = self.file_pattern.findall(query)
        slots["file_path"] = file_matches[0] if file_matches else None
        img_matches = self.image_pattern.findall(query)
        slots["image_path"] = img_matches[0] if img_matches else None
        q = query.lower()
        for lang in ["英文", "中文", "日文", "韩文", "法文", "德文"]:
            if lang in q:
                slots["language"] = lang
                break
        if any(kw in q for kw in ["最近", "刚才", "最新"]):
            slots["time_range"] = "recent"
        elif "今天" in q:
            slots["time_range"] = "today"
        elif "昨天" in q:
            slots["time_range"] = "yesterday"
        for fmt in ["json", "csv", "markdown", "md格式", "表格"]:
            if fmt in q:
                slots["format"] = fmt.replace("md格式", "markdown").replace("表格", "table")
                break
        return slots


# ================================================================
#  8. 训练模板（训练与联盟评估共用）
# ================================================================
TASK_TEMPLATES: Dict[IntentType, List[str]] = {
    IntentType.ADD_FILE: [
        "添加文件 {f} 内容为 {c}", "创建新文件 {f}", "写入 {f} 保存内容: {c}",
        "新建一个 {f}", "帮我写一个 {f}", "把以下内容保存到 {f}",
        "生成文件 {f}", "帮我创建 {f} 然后写入内容",
    ],
    IntentType.DELETE_FILE: [
        "删除文件 {f}", "移除 {f}", "清理文件 {f}",
        "把 {f} 删掉", "帮我删除 {f}", "移除掉 {f}",
        "清理掉 {f}", "把 {f} 从项目中删去",
    ],
    IntentType.MODIFY_FILE: [
        "修改文件 {f} 内容为 {c}", "更新 {f} 改成 {c}",
        "编辑 {f}", "帮我改一下 {f}", "把 {f} 的内容替换一下",
        "更新 {f} 的内容", "修改 {f} 中的配置",
    ],
    IntentType.REFACTOR: [
        "整理文件夹", "重构项目结构", "把文件按类型分类归档",
        "帮我整理一下目录", "重命名文件", "移动文件到对应文件夹",
        "代码重构", "优化项目结构",
    ],
    IntentType.QUERY: [
        "查询文件 {f} 里写了什么", "搜索 {f} 的内容", "{f} 存在吗",
        "帮我找一下 {f}", "查看 {f}", "{f} 里面有什么",
        "搜索强化学习", "查一下这个文件", "找到红警3的启动程序在哪里",
    ],
    IntentType.DEEP_SEARCH: [
        "对比分析 {f} 和 {g}", "深度分析 {f} 的趋势与影响",
        "综合分析 {f} 和 {g} 的差异", "帮我比较一下 {f} 和 {g}",
        "分析 {f} 的优缺点", "深入研究 {f} 的发展方向",
        "评估 {f} 和 {g} 的区别",
    ],
    IntentType.VAGUE: [
        "你好", "嗨", "在吗", "你是谁", "帮我一下",
        "请问", "嗯", "好的", "谢谢", "哈哈",
        "今天天气怎么样", "讲个笑话", "随便聊聊",
        "你能做什么", "介绍一下自己", "无聊",
        "123", "test", "hello", "hi", "hey",
        "你是AI吗", "我有个问题", "等等",
    ],
    IntentType.FILE_ANALYSIS: [
        "只读分析 {f} 的代码质量", "审查 {f} 的结构",
        "检查 {f} 中的问题", "诊断 {f} 的性能",
        "评估 {f} 的复杂度", "分析 {f} 的依赖关系",
        "深度分析 {f} 的架构", "审查项目的代码规范",
    ],
    IntentType.FILE_SCAN: [
        "扫描项目中的所有文件", "遍历目录列出所有文件",
        "统计项目文件数量", "列举 {f} 里的内容",
        "扫描代码中的TODO标记", "检查项目结构",
        "列出所有配置文件", "扫描重复文件",
    ],
    IntentType.REPORT_GENERATE: [
        "生成代码审查报告", "输出项目分析报告",
        "写一份 {f} 的总结报告", "创建JSON格式的分析结果",
        "汇总生成CSV报告", "生成项目质量报告",
        "输出代码复杂度报告", "生成依赖分析报告",
    ],
    IntentType.FILENAME_QUERY: [
        "为我找到最近添加的文件名", "最近修改的文件有哪些",
        "查找最近创建的文件", "最近编辑过的文件名",
        "最新添加的文件有哪些", "最近保存的文件名列表",
        "搜索最近的txt文件", "最近修改的py文件有哪些",
        "找到本周创建的文件", "最近一天修改的文件",
        "recently modified files", "find newly created files",
        "最新的文件名有哪些", "最近操作过的文件",
    ],
}
SAMPLE_FILES = ["notes.txt", "config.yaml", "main.py", "app.json",
                "report.md", "readme.md", "data.csv"]
SAMPLE_CONTENT = "这是一段用于演示的文档内容。"

# 6 路投票的源名称（与 agent_rl.AgentRL 的 6 路一一对应但内容不同）
SOURCE_NAMES = ("llm", "policy", "league", "rule", "retrieval", "single")


# ================================================================
#  9. MetaAgent —— 协调器（中央训练、分布执行）
# ================================================================
class MetaAgent:
    """多 Agent 协调器：6 路仲裁 + MAPPO 联盟训练 + 集成策略推断。"""

    FEATURE_DIM = OBS_DIM
    CONFIDENCE_THRESHOLD = 0.65

    def __init__(self, model_path: str = None, single_cross_vote: bool = True):
        # ---- 个体策略（不同初始化 / 超参数 / 观测）----
        self.intent_agent = IntentPolicy(INTENT_HP)
        self.subintent_agent = SubIntentPolicy(SUBINTENT_HP)
        self.slot_agent = SlotPolicy(SLOT_HP)
        self._agents: Dict[str, PolicyAgent] = {
            "intent": self.intent_agent,
            "subintent": self.subintent_agent,
            "slot": self.slot_agent,
        }

        # ---- 共享组件 ----
        self.feature_extractor = _SemanticFeatureExtractor()
        self.query_rewriter = QueryRewriter()
        self.llm_classifier = LLMIntentClassifier()
        self.intent_retriever = get_intent_retriever()
        # Engram 记忆读写的是策略观测（OBS_DIM 维）
        self.engram = EngramConditionalMemory(d_model=128, input_dim=OBS_DIM)

        # ---- 联盟 / 对手 ----
        self.league = League(lambda: TransformerIntentQNetwork(
            OBS_DIM, 128, len(IntentPolicy.INTENT_LIST), num_loops=4))
        if not len(self.league):
            # 播种基线成员：使第 6 路中的 league 路自启动即存在；
            # 基线 = 当前初始策略的副本，只有训练后的策略胜过它才会晋升，
            # 之后联盟由历史快照构成（多样性随训练增长）。
            self.league.promote(self.intent_agent.q_network,
                                win_rate=0.5, step=0, hp_name="intent")
        self.query_opponent = QueryOpponent()
        self._opp_player: Optional[LeaguePlayer] = None
        self._opp_active = False

        # ---- 协调：源权重（EXP3 在线学习，合作场景下多样性必须配合协调机制）----
        self.source_weights: Dict[str, float] = {n: 1.0 for n in SOURCE_NAMES}
        self._last_source_scores: Dict[str, Tuple[IntentType, float]] = {}
        self._last_votes: Dict[str, IntentType] = {}
        self._last_final: Optional[IntentType] = None
        self._last_intent_vote: Optional[IntentType] = None
        self._last_confidence = 0.5
        self._last_local: Dict[str, float] = {}
        self._pending: Dict[str, Dict[str, Any]] = {}

        # ---- 学习状态 ----
        self.reward_scale = RunningScale()
        self.replay_buffer: List[Transition] = []
        self.training_step = 0
        self._steps_since_update = 0
        self._distill_bank: deque = deque(maxlen=512)
        self.epsilon = 0.30              # 与单 Agent (0.3/0.02/0.998) 刻意不同
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.996
        self.gamma = INTENT_HP.gamma
        self.intent_history: List[Tuple[int, float]] = []
        self.reward_history: List[float] = []
        self.last_llm_target = None
        self.last_llm_sub_intent = None
        self.last_llm_slots: Dict[str, str] = {}

        # 第 6 路交叉引用的单 Agent 神经投票器（惰性、无 LLM、不触发第 6 路）
        self.single_cross_vote = single_cross_vote
        self._single_voter: Optional[AgentRL] = None

        self.model_path = model_path or DEFAULT_MARL_MODEL_PATH
        self.loaded_from_disk = False
        self._auto_load()

    # ---------------------------------------------------------- 持久化
    @property
    def single_voter(self) -> Optional[AgentRL]:
        if self._single_voter is None and self.single_cross_vote:
            self._single_voter = AgentRL(marl_vote=False)
        return self._single_voter

    def _auto_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            ok = self.load(self.model_path)
            self.loaded_from_disk = bool(ok)
            if ok:
                print(f"\n  [MARL] 已加载模型 {self.model_path} "
                      f"(ε={self.epsilon:.4f}, 联盟={len(self.league)}, 步数={self.training_step})")
        except Exception as e:
            self.loaded_from_disk = False
            print(f"  [MARL] 加载失败({e})，重新初始化。")

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "format": MARL_CKPT_FORMAT,
            "intent_q": self.intent_agent.q_network.state_dict(),
            "intent_opt": self.intent_agent.optimizer.state_dict(),
            "critic_intent": self.intent_agent.critic.state_dict(),
            "critic_intent_opt": self.intent_agent.critic_optimizer.state_dict(),
            "subintent_q": self.subintent_agent.q_network.state_dict(),
            "subintent_opt": self.subintent_agent.optimizer.state_dict(),
            "critic_subintent": self.subintent_agent.critic.state_dict(),
            "critic_subintent_opt": self.subintent_agent.critic_optimizer.state_dict(),
            "slot_net": self.slot_agent.q_network.state_dict(),
            "slot_opt": self.slot_agent.optimizer.state_dict(),
            "critic_slot": self.slot_agent.critic.state_dict(),
            "critic_slot_opt": self.slot_agent.critic_optimizer.state_dict(),
            "engram": self.engram.state_dict(),
            "league": self.league.state_dict(),
            "query_opponent": self.query_opponent.state_dict(),
            "source_weights": dict(self.source_weights),
            "reward_scale": self.reward_scale.state_dict(),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
        }, path)
        print(f"  [MARL] 已保存 → {path}")

    def load(self, path: str) -> bool:
        """加载检查点。格式不匹配（旧版 DQN 检查点）时返回 False 并保持新初始化。"""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if int(ckpt.get("format", 0)) != MARL_CKPT_FORMAT:
            print("  [MARL] 旧版检查点与新架构(272 维观测 / MAPPO)不兼容，忽略旧权重。")
            return False

        def _load_nn(module, key):
            if key in ckpt:
                try:
                    module.load_state_dict(ckpt[key])
                    return True
                except Exception:
                    return False
            return False

        _load_nn(self.intent_agent.q_network, "intent_q")
        _load_nn(self.intent_agent.critic, "critic_intent")
        _load_nn(self.subintent_agent.q_network, "subintent_q")
        _load_nn(self.subintent_agent.critic, "critic_subintent")
        _load_nn(self.slot_agent.q_network, "slot_net")
        _load_nn(self.slot_agent.critic, "critic_slot")
        _load_nn(self.engram, "engram")
        for agent, qk, ck, ok, cok in (
            (self.intent_agent, "intent_q", "critic_intent", "intent_opt", "critic_intent_opt"),
            (self.subintent_agent, "subintent_q", "critic_subintent", "subintent_opt", "critic_subintent_opt"),
            (self.slot_agent, "slot_net", "critic_slot", "slot_opt", "critic_slot_opt"),
        ):
            if qk in ckpt:
                try:
                    agent.optimizer.load_state_dict(ckpt[ok])
                    # 超参数以代码为准（检查点里是旧 lr）
                    for g in agent.optimizer.param_groups:
                        g["lr"] = agent.hp.lr
                except Exception:
                    pass
            if ck in ckpt:
                try:
                    agent.critic_optimizer.load_state_dict(ckpt[cok])
                    for g in agent.critic_optimizer.param_groups:
                        g["lr"] = agent.hp.critic_lr
                except Exception:
                    pass
        if "league" in ckpt:
            try:
                self.league.load_state_dict(ckpt["league"])
            except Exception:
                pass
        if "query_opponent" in ckpt:
            self.query_opponent.load_state_dict(ckpt["query_opponent"])
        if "source_weights" in ckpt:
            self.source_weights.update({k: float(v) for k, v in ckpt["source_weights"].items()})
        if "reward_scale" in ckpt:
            self.reward_scale.load_state_dict(ckpt["reward_scale"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon))
        self.training_step = int(ckpt.get("training_step", 0))
        return True

    # ---------------------------------------------------------- 观测
    def _base_sem(self, query: str) -> np.ndarray:
        return self.feature_extractor.extract(query)

    def _intent_aux(self, critic_unc: float) -> np.ndarray:
        """意图 Agent 的辅助通道：联盟对手上下文（对手建模 → 处理非平稳性）。"""
        if self._opp_active and self._opp_player is not None:
            strength = float(self._opp_player.win_rate)
            opp_id = float((self._opp_player.pid % 89) / 89.0)
        else:
            strength, opp_id = 0.5, 0.5      # 中性/自对弈上下文
        league_n = min(len(self.league) / max(MAX_LEAGUE, 1), 1.0)
        return np.array([strength, opp_id, league_n, float(np.clip(critic_unc, 0.0, 1.0))],
                        dtype=np.float32)

    def _subintent_aux(self, query: str, sem: np.ndarray) -> np.ndarray:
        cached = self._ret_cache(query)
        ret_conf = float(cached[1]) if cached and cached[0] is not None else 0.0
        rule_hit = 1.0 if self.subintent_agent.rule_predict(query) is not None else 0.0
        return np.array([ret_conf,
                         min(len(query) / 50.0, 1.0),
                         rule_hit,
                         float(self.subintent_agent.reverse_map.__len__()) / 20.0],
                        dtype=np.float32)

    def _slot_aux(self, query: str) -> np.ndarray:
        has_file = 1.0 if self.slot_agent.file_pattern.search(query) else 0.0
        has_img = 1.0 if self.slot_agent.image_pattern.search(query) else 0.0
        has_lang = 1.0 if any(l in query for l in
                              ("英文", "中文", "日文", "韩文", "法文", "德文")) else 0.0
        has_time = 1.0 if any(k in query for k in ("最近", "今天", "昨天", "本周")) else 0.0
        return np.array([has_file, has_img, has_lang, has_time], dtype=np.float32)

    def _ret_cache(self, query: str):
        """检索路结果缓存（观测与投票共用，避免重复向量检索）。"""
        if getattr(self, "_ret_query", None) == query:
            return self._ret_value
        self._ret_query = query
        try:
            self._ret_value = self.intent_retriever.vote(query)
        except Exception:
            self._ret_value = (None, 0.0, None)
        return self._ret_value

    def _obs(self, role: str, query: str, state: AgentState,
             sem: np.ndarray = None, critic_unc: float = None) -> np.ndarray:
        sem = self._base_sem(query) if sem is None else sem
        state_vec = state.get_state_vector()
        if role == "intent":
            aux = self._intent_aux(0.5 if critic_unc is None else critic_unc)
        elif role == "subintent":
            aux = self._subintent_aux(query, sem)
        else:
            aux = self._slot_aux(query)
        return self._agents[role].encode(sem, state_vec, aux)

    def _extract_features(self, query: str, state: AgentState) -> np.ndarray:
        """兼容接口：单 Agent 风格的 268 维基础特征（仅供外部/诊断使用）。"""
        return np.concatenate([self._base_sem(query),
                               state.get_state_vector()]).astype(np.float32)

    def _central_context(self, query: str, state: AgentState,
                         stats: Dict[str, float]) -> np.ndarray:
        """集中式 Critic 的团队上下文（CTDE：中央训练用，执行时本地观测）。"""
        sv = state.get_state_vector()
        ctx = [
            stats.get("agreement", 0.0), stats.get("pool_entropy", 0.0),
            stats.get("margin", 0.0), stats.get("llm_conf", 0.0),
            stats.get("rule_conf", 0.0), stats.get("ret_conf", 0.0),
            stats.get("single_conf", 0.0), stats.get("policy_conf", 0.0),
            stats.get("league_conf", 0.0), stats.get("final_conf", 0.0),
            sv[1], sv[4], sv[5], sv[9],
            float(min(len(self.league) / max(MAX_LEAGUE, 1), 1.0)),
            float(np.clip(self.epsilon, 0.0, 1.0)),
        ]
        return np.asarray(ctx, dtype=np.float32)

    # ---------------------------------------------------------- 集成策略
    def _critic_uncertainty(self, obs: np.ndarray, ctx: np.ndarray) -> float:
        try:
            with torch.no_grad():
                x = torch.FloatTensor(np.concatenate([obs, ctx])).unsqueeze(0)
                _, std = self.intent_agent.critic(x, "min")
            return float(std.item())
        except Exception:
            return 0.0

    @torch.no_grad()
    def _policy_probs(self, obs: np.ndarray, include_league: bool = True
                      ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """当前策略 + 联盟集成（按胜率加权），返回 (混合概率, 联盟概率, JS 分歧)。"""
        cur = self.intent_agent.probs_np(obs)
        lg = self.league.aggregate_probs(obs) if (include_league and len(self.league)) else None
        if lg is None:
            return cur, None, 0.0
        mixed = 0.6 * cur + 0.4 * lg
        js = _js_divergence(cur, lg) / math.log(2.0)
        return mixed, lg, float(np.clip(js, 0.0, 1.0))

    # ---------------------------------------------------------- 第 6 路投票
    def vote(self, query: str, state: AgentState
             ) -> Tuple[Optional[IntentType], float, Dict[str, Any]]:
        """供 agent_rl 第 6 路调用：只读、无副作用、不调用 LLM。

        返回 (意图, 校准置信度, 诊断信息)。置信度 = 集成策略 top 概率 ×
        (1 - 成员分歧)，分歧即不确定性估计。
        """
        obs = self._obs("intent", query, state)
        mixed, lg, js = self._policy_probs(obs, include_league=True)
        idx = int(mixed.argmax())
        base = float(mixed[idx])
        conf = base * (0.55 + 0.45 * (1.0 - js))
        info = {
            "base_conf": base,
            "disagreement": js,
            "entropy": float(-(mixed * np.log(np.clip(mixed, 1e-9, 1.0))).sum()),
            "league_size": len(self.league),
            "sources": ["policy", "league"] if lg is not None else ["policy"],
        }
        return self.intent_agent.action_map[idx], float(np.clip(conf, 0.0, 1.0)), info

    # ---------------------------------------------------------- 6 路仲裁
    def _source_weight(self, name: str) -> float:
        return self.source_weights.get(name, 1.0)

    @property
    def last_sources(self) -> List[str]:
        """最近一次仲裁实际参与的来源（llm/policy/league/rule/retrieval/single）。"""
        return sorted(self._last_source_scores)

    def classify_intent(self, query: str, state: AgentState,
                        use_llm: bool = True) -> Tuple[IntentType, float]:
        """6 路仲裁：LLM / 当前策略 / 联盟策略 / 规则 / 检索 / 单Agent神经路。"""
        self.last_llm_target = None
        self.last_llm_sub_intent = None
        self.last_llm_slots = {}
        return self._arbitrate(query, state, use_llm=use_llm,
                              include_single=self.single_cross_vote, explore=True)

    def _arbitrate(self, query: str, state: AgentState, use_llm: bool,
                   include_single: bool, explore: bool,
                   collect: bool = True) -> Tuple[IntentType, float]:
        stats: Dict[str, float] = {}
        candidates: List[Tuple[IntentType, float, str]] = []
        votes: Dict[str, IntentType] = {}

        # --- 1) LLM ---
        if use_llm:
            llm_result = self.llm_classifier.classify(query)
            if llm_result:
                intent_value, target, confidence, sub_intent, slots = llm_result
                try:
                    intent = IntentType(intent_value)
                    conf = float(np.clip(confidence, 0.0, 1.0))
                    w = self._source_weight("llm")
                    candidates.append((intent, min(1.0, conf * 1.2) * w, "llm"))
                    votes["llm"] = intent
                    stats["llm_conf"] = conf
                    self.last_llm_target = target
                    self.last_llm_sub_intent = sub_intent
                    self.last_llm_slots = slots
                except ValueError:
                    pass

        # --- 2) 当前策略（意图 Agent，含探索采样）---
        sem = self._base_sem(query)
        obs = self._obs("intent", query, state, sem=sem)
        ctx = self._central_context(query, state, stats)
        value, unc = 0.0, 0.0
        try:
            with torch.no_grad():
                x = torch.FloatTensor(np.concatenate([obs, ctx])).unsqueeze(0)
                value, std = self.intent_agent.critic(x, "min")
            value, unc = float(value.item()), float(std.item())
        except Exception:
            pass

        action, logprob, entropy, probs = self.intent_agent.act(obs, explore=explore)
        policy_intent = self.intent_agent.action_map[int(action)]
        policy_conf = float(probs.max())
        candidates.append((policy_intent,
                           policy_conf * self._source_weight("policy"), "policy"))
        votes["policy"] = policy_intent
        stats["policy_conf"] = policy_conf
        stats["policy_entropy"] = entropy / math.log(len(self.intent_agent.INTENT_LIST) + 1e-9)
        self._last_intent_vote = policy_intent
        self.intent_history.append((int(action), policy_conf))

        # --- 3) 联盟集成策略 ---
        league_probs = self.league.aggregate_probs(obs) if len(self.league) else None
        if league_probs is not None:
            lg_intent = self.intent_agent.action_map[int(league_probs.argmax())]
            lg_conf = float(league_probs.max())
            candidates.append((lg_intent,
                               lg_conf * self._source_weight("league"), "league"))
            votes["league"] = lg_intent
            stats["league_conf"] = lg_conf

        # --- 4) 规则 ---
        rule_intent = self.intent_agent.rule_predict(query)
        if rule_intent is not None:
            rule_weight = 0.55 if use_llm else 0.85
            candidates.append((rule_intent,
                               rule_weight * self._source_weight("rule"), "rule"))
            votes["rule"] = rule_intent
            stats["rule_conf"] = rule_weight

        # --- 5) 检索 ---
        ret_intent, ret_conf, _ = self._ret_cache(query)
        if ret_intent is not None:
            ret_weight = 0.50 if use_llm else 0.75
            candidates.append((ret_intent,
                               ret_conf * ret_weight * self._source_weight("retrieval"),
                               "retrieval"))
            votes["retrieval"] = ret_intent
            stats["ret_conf"] = float(ret_conf)

        # --- 6) 单 Agent 神经路（交叉多样性；无 LLM、不会回环到第 6 路）---
        if include_single and self.single_voter is not None:
            try:
                s_intent, s_conf, _ = self.single_voter.neural_vote(query, state)
                if s_intent is not None:
                    candidates.append((s_intent,
                                       float(s_conf) * 0.45 * self._source_weight("single"),
                                       "single"))
                    votes["single"] = s_intent
                    stats["single_conf"] = float(s_conf)
            except Exception:
                pass

        # --- 聚合：加权得分 / 每意图取均值 ---
        scores: Dict[IntentType, Dict] = {}
        source_scores: Dict[str, Tuple[IntentType, float]] = {}
        for intent, score, source in candidates:
            scores.setdefault(intent, {"score": 0.0, "sources": []})
            scores[intent]["score"] += score
            scores[intent]["sources"].append(source)
            source_scores[source] = (intent, score)
        for v in scores.values():
            v["score"] /= len(v["sources"])

        final_intent = max(scores, key=lambda k: scores[k]["score"])
        final_conf = float(scores[final_intent]["score"])

        ranked = sorted((v["score"] for v in scores.values()), reverse=True)
        stats["margin"] = float(ranked[0] - ranked[1]) if len(ranked) > 1 else float(ranked[0])
        top = np.zeros(len(self.intent_agent.INTENT_LIST), dtype=np.float32)
        for it, v in scores.items():
            top[self.intent_agent.reverse_map[it]] = v["score"]
        s = top.sum() + 1e-8
        stats["pool_entropy"] = float(-(top / s * np.log(top / s + 1e-9)).sum()
                                     / math.log(len(top)))
        stats["agreement"] = float(len(set(votes.values())) <= 2) if votes else 0.0
        stats["final_conf"] = final_conf

        # --- 记录（供信用分配 / 协调权重学习使用）---
        self._last_source_scores = dict(source_scores)
        self._last_votes = dict(votes)
        self._last_final = final_intent

        if collect:
            self._pending["intent"] = {
                "obs": obs, "ctx": ctx, "action": int(action),
                "logprob": logprob, "value": value,
                "opp_pred": self._opp_predict(obs),
            }
            self._distill_bank.append(obs)

        if final_conf < self.CONFIDENCE_THRESHOLD:
            final_intent, final_conf = self._engram_assist(obs, final_intent, final_conf)

        final_intent = self._semantic_conflict_guard(query, final_intent, scores)
        self._last_confidence = final_conf
        return final_intent, final_conf

    @torch.no_grad()
    def _opp_predict(self, obs: np.ndarray) -> int:
        """联盟对手在同一观测上的预测（head-to-head 胜负信号）。"""
        if not (self._opp_active and self._opp_player is not None):
            return -1
        try:
            x = torch.FloatTensor(obs).unsqueeze(0)
            return int(self._opp_player.net(x).argmax(dim=1).item())
        except Exception:
            return -1

    def _engram_assist(self, obs: np.ndarray, intent: IntentType,
                       conf: float) -> Tuple[IntentType, float]:
        """低置信度时用条件记忆增强（与单 Agent 同机制，但作用在策略观测上）。"""
        try:
            with torch.no_grad():
                feat = torch.FloatTensor(obs).unsqueeze(0)
                cls_hidden = self.intent_agent.q_network.transformer(feat)[:, 0]
                conf_t = torch.FloatTensor([[conf]])
                enhanced = self.engram.read(cls_hidden, feat, conf_t)
                enhanced_q = self.intent_agent.q_network.head(enhanced)
                probs = F.softmax(enhanced_q, dim=-1).squeeze(0).numpy()
                eng_action = int(probs.argmax())
                eng_conf = float(probs[eng_action])
                if eng_conf > conf + 0.1:
                    return self.intent_agent.action_map[eng_action], eng_conf
        except Exception:
            pass
        return intent, conf

    def _semantic_conflict_guard(self, query: str, proposed: IntentType,
                                  votes: Dict[IntentType, Dict]) -> IntentType:
        """语义冲突护栏（只读/查询类关键词下禁止落进写操作意图）。"""
        q = query.lower().strip()

        search_kws = ["找", "查", "搜索", "查询", "检索", "怎么", "什么", "哪里",
                       "search", "find", "查看", "看看", "在哪", "定位",
                       "识别", "截屏", "截图", "ocr", "文字", "输出", "提取", "读取"]
        has_search = any(kw in q for kw in search_kws)
        delete_kws = ["删除", "移除", "清理", "删掉", "干掉", "remove", "delete"]
        has_delete_kw = any(kw in q for kw in delete_kws)

        if has_search and not has_delete_kw:
            if proposed in {IntentType.DELETE_FILE, IntentType.REFACTOR,
                            IntentType.ADD_FILE, IntentType.MODIFY_FILE}:
                return IntentType.QUERY

        readonly_kws = ["只读", "不修改", "不移动", "不删除", "不重命名", "不覆盖"]
        has_readonly = any(kw in q for kw in readonly_kws)
        analysis_kws = ["分析", "审查", "检查", "诊断", "评估"]
        has_analysis = any(kw in q for kw in analysis_kws)
        scan_kws = ["扫描", "遍历", "列举", "统计"]
        has_scan = any(kw in q for kw in scan_kws)
        report_kws = ["报告", "生成报告", "输出报告"]
        has_report = any(kw in q for kw in report_kws)

        if has_readonly or has_analysis or has_scan or has_report:
            if proposed in {IntentType.ADD_FILE, IntentType.DELETE_FILE,
                            IntentType.MODIFY_FILE, IntentType.REFACTOR}:
                if has_analysis:
                    return IntentType.FILE_ANALYSIS
                if has_scan:
                    return IntentType.FILE_SCAN
                if has_report:
                    return IntentType.REPORT_GENERATE
                return IntentType.QUERY

        filename_query_kws = ["最近添加的文件", "最近修改的文件", "最近创建的文件",
                              "最近的文件", "最新文件", "文件名搜索", "搜索文件名"]
        if any(kw in q for kw in filename_query_kws):
            if proposed in {IntentType.DELETE_FILE, IntentType.MODIFY_FILE,
                            IntentType.ADD_FILE, IntentType.REFACTOR}:
                return IntentType.FILENAME_QUERY
        return proposed

    # ---------------------------------------------------------- 子任务
    def get_sub_intent(self, query: str, features: np.ndarray = None) -> QuerySubIntent:
        rule_result = self.subintent_agent.rule_predict(query)
        if rule_result is not None:
            self._last_local["subintent"] = 1.0 if (
                self._last_votes.get("single") is not None) else 0.5
            return rule_result

        obs = self._obs("subintent", query, self._cur_state)
        action, logprob, _, probs = self.subintent_agent.act(obs, explore=True)
        sub = self.subintent_agent.action_map[int(action)]
        if self.single_cross_vote and self.single_voter is not None:
            pass
        self._pending["subintent"] = {
            "obs": obs, "ctx": self._central_context(query, self._cur_state, {}),
            "action": int(action), "logprob": logprob, "value": 0.0, "opp_pred": -1,
        }
        self._last_local["subintent"] = 0.0
        return sub

    def get_slots(self, query: str, features: np.ndarray = None) -> Dict[str, Any]:
        obs = self._obs("slot", query, self._cur_state)
        action, logprob, _, _ = self.slot_agent.act(obs, explore=True)
        neural_slots = self.slot_agent.predict(obs)
        rule_slots = self.slot_agent.rule_extract(query)

        self._pending["slot"] = {
            "obs": obs, "ctx": self._central_context(query, self._cur_state, {}),
            "action": action, "logprob": logprob, "value": 0.0, "opp_pred": -1,
        }
        # 局部信用：神经预测与规则抽取的一致率（自监督）
        agree, total = 0, 0
        for key in ("language", "time_range", "format"):
            if rule_slots.get(key) is not None:
                total += 1
                if neural_slots.get(key, (None, 0.0))[0] == rule_slots[key]:
                    agree += 1
        self._last_local["slot"] = (2.0 * agree / total - 1.0) if total else 0.0

        final_slots: Dict[str, Any] = {}
        for key in ["file_path", "image_path", "language", "time_range", "format"]:
            rule_val = rule_slots.get(key)
            neural_val, neural_conf = neural_slots.get(key, (None, 0.0))
            if rule_val is not None:
                final_slots[key] = rule_val
            elif key in ("file_path", "image_path"):
                continue        # 占位标记不能当路径用（交给 query_rewriter 提取）
            elif neural_conf > 0.6 and neural_val != "absent":
                final_slots[key] = neural_val
        return final_slots

    def process(self, query: str, state: AgentState
                ) -> Tuple[IntentType, Any, float, List[str]]:
        """完整 MARL 管线：改写 → 6 路意图仲裁 → 子意图 → 槽位 → 澄清。"""
        rewritten, entities, relations = self.query_rewriter.rewrite(query)
        state.query_rewrite_history.append(rewritten)
        self._cur_state = state

        intent, confidence = self.classify_intent(query, state)
        routing_path = self._get_routing_path(intent)

        sub_intent = None
        if intent == IntentType.QUERY:
            if self.last_llm_sub_intent:
                try:
                    sub_intent = QuerySubIntent(self.last_llm_sub_intent)
                except ValueError:
                    sub_intent = self.get_sub_intent(query)
            else:
                sub_intent = self.get_sub_intent(query)
            state.current_sub_intent = sub_intent
            routing_path = self._get_query_routing_path(sub_intent)

        slots = self.get_slots(query)
        if self.last_llm_slots:
            slots.update(self.last_llm_slots)
        state.slots = slots

        clarifications = []
        if intent == IntentType.VAGUE:
            clarifications = self.query_rewriter.generate_clarification(query)
            if not clarifications or not should_clarify(confidence, state):
                intent = IntentType.QUERY
                routing_path = self._get_routing_path(intent)
                confidence = max(confidence, 0.4)
        elif intent in {IntentType.ADD_FILE, IntentType.DELETE_FILE, IntentType.MODIFY_FILE}:
            target = self.last_llm_target or self.query_rewriter.extract_target(query)
            if not target:
                clarifications = self.query_rewriter.generate_clarification(query)

        return intent, routing_path, confidence, clarifications

    def _get_routing_path(self, intent: IntentType):
        from agent_state import RoutingPath
        routing_map = {
            IntentType.ADD_FILE: RoutingPath.FILE_ADD,
            IntentType.DELETE_FILE: RoutingPath.FILE_DELETE,
            IntentType.MODIFY_FILE: RoutingPath.FILE_MODIFY,
            IntentType.REFACTOR: RoutingPath.FILE_REFACTOR,
            IntentType.QUERY: RoutingPath.SINGLE_VECTOR,
            IntentType.DEEP_SEARCH: RoutingPath.DEEP_SEARCH,
            IntentType.VAGUE: RoutingPath.CLARIFICATION,
            IntentType.FILE_ANALYSIS: RoutingPath.FILE_ANALYSIS,
            IntentType.FILE_SCAN: RoutingPath.FILE_SCAN,
            IntentType.REPORT_GENERATE: RoutingPath.REPORT_GENERATE,
            IntentType.FILENAME_QUERY: RoutingPath.FILENAME_QUERY,
        }
        return routing_map[intent]

    def _get_query_routing_path(self, sub_intent: QuerySubIntent):
        from agent_state import RoutingPath
        routing_map = {
            QuerySubIntent.QUERY_METADATA: RoutingPath.QUERY_METADATA,
            QuerySubIntent.QUERY_CONTENT: RoutingPath.QUERY_CONTENT,
            QuerySubIntent.QUERY_OCR: RoutingPath.QUERY_OCR,
            QuerySubIntent.QUERY_SUMMARY: RoutingPath.QUERY_SUMMARY,
            QuerySubIntent.QUERY_TRANSLATE: RoutingPath.QUERY_TRANSLATE,
            QuerySubIntent.QUERY_EXTRACT: RoutingPath.QUERY_EXTRACT,
            QuerySubIntent.QUERY_SEARCH: RoutingPath.SINGLE_VECTOR,
            QuerySubIntent.QUERY_SYSTEM: RoutingPath.QUERY_SYSTEM,
            QuerySubIntent.QUERY_OTHER: RoutingPath.SINGLE_VECTOR,
            QuerySubIntent.QUERY_FILE_SCAN: RoutingPath.FILE_SCAN,
            QuerySubIntent.QUERY_FILE_ANALYZE: RoutingPath.FILE_ANALYSIS,
            QuerySubIntent.QUERY_READONLY_REPORT: RoutingPath.REPORT_GENERATE,
        }
        return routing_map.get(sub_intent, RoutingPath.SINGLE_VECTOR)

    # ---------------------------------------------------------- 信用分配
    def _marginal_contribution(self) -> float:
        """差分奖励的边际贡献：去掉「当前策略」这一票后共识是否翻转。"""
        final = self._last_final
        votes = {k: v for k, v in self._last_votes.items() if k != "policy"}
        if final is None or not votes:
            return 1.0
        scores: Dict[IntentType, float] = {}
        for src, it in votes.items():
            scores[it] = scores.get(it, 0.0) + self._source_weight(src)
        winner_wo = max(scores, key=scores.get)
        return 1.0 if winner_wo != final else 0.0

    def _assign_credit(self, team_reward: float) -> Dict[str, float]:
        """按边际贡献把团队奖励分给各 Agent，并叠加协调/局部奖励。

        r_intent = (0.6 + 0.4 * marginal) * R̂ + coord_bonus * 一致(±) + h2h * 胜负
        r_sub    = 0.45 * R̂ + local_sub（与 LLM/规则子意图一致度）
        r_slot   = 0.45 * R̂ + local_slot（神经槽位与规则一致度）
        R̂ 为尺度归一化后的团队奖励。
        """
        r_hat = self.reward_scale.scale(team_reward)
        marg = self._marginal_contribution()
        coord = COORD_BONUS * (1.0 if (self._last_intent_vote is not None
                                       and self._last_final is not None
                                       and self._last_intent_vote == self._last_final)
                              else -0.5)
        h2h = 0.0
        pend = self._pending.get("intent")
        if pend is not None and team_reward > 0 and pend.get("opp_pred", -1) >= 0:
            mine = self.intent_agent.action_map.get(pend["action"])
            opp = self.intent_agent.action_map.get(pend["opp_pred"])
            if mine == self._last_final and opp != self._last_final:
                h2h = H2H_BONUS
            elif mine != self._last_final and opp == self._last_final:
                h2h = -H2H_BONUS

        credit = {
            "intent": r_hat * (0.6 + 0.4 * marg) + coord + h2h,
            "subintent": 0.45 * r_hat + 0.6 * self._last_local.get("subintent", 0.0),
            "slot": 0.45 * r_hat + 0.6 * self._last_local.get("slot", 0.0),
        }
        return credit

    def _update_source_weights(self, team_reward: float):
        """EXP3/Hedge 协调器：按各路与最终共识的一致性好坏在线调整源权重。"""
        if not self._last_source_scores or self._last_final is None:
            return
        good = team_reward > 0
        for name, (intent, _) in self._last_source_scores.items():
            if name not in self.source_weights:
                continue
            agreed = (intent == self._last_final)
            r = (1.0 if agreed else -0.5) if good else (-1.0 if agreed else 0.5)
            self.source_weights[name] = max(
                self.source_weights[name] * math.exp(EXP3_LR * r), EXP3_FLOOR)
        mean = sum(self.source_weights.values()) / len(self.source_weights)
        if mean > 0:
            for k in self.source_weights:
                self.source_weights[k] /= mean

    # ---------------------------------------------------------- 在线学习
    def learn(self, reward: float, state: AgentState, done: bool):
        """接收团队奖励 → 信用分配 → 写入 rollout → 触发 MAPPO 更新。"""
        state.add_reward(reward)
        self.reward_history.append(reward)

        if not self._pending:
            self._maybe_update()
            return

        query = state.dialog_history[-1].user_query if state.dialog_history else ""
        self.reward_scale.update(reward)
        credit = self._assign_credit(reward)

        for role, p in list(self._pending.items()):
            agent = self._agents.get(role)
            if agent is None:
                continue
            try:
                next_obs = self._obs(role, query, state)
                next_ctx = self._central_context(query, state, {})
            except Exception:
                continue
            self.replay_buffer.append(Transition(
                agent=role, obs=np.asarray(p["obs"], dtype=np.float32),
                ctx=np.asarray(p["ctx"], dtype=np.float32),
                action=p["action"], logprob=float(p["logprob"]),
                reward=float(credit.get(role, 0.0)), done=bool(done),
                value=float(p.get("value", 0.0)),
                next_obs=next_obs, next_ctx=next_ctx,
                opp_pred=int(p.get("opp_pred", -1)),
            ))
        self._pending.clear()

        # Engram 写入：高置信度时积累经验
        if self._last_confidence > 0.7 and "intent" in self._last_votes:
            try:
                intent_idx = self.intent_agent.reverse_map[self._last_final]
                with torch.no_grad():
                    obs = np.asarray(self.replay_buffer[-1].obs)
                    feat = torch.FloatTensor(obs).unsqueeze(0)
                    cls_hidden = self.intent_agent.q_network.transformer(feat)[:, 0]
                    conf_t = torch.FloatTensor([[self._last_confidence]])
                    self.engram.write(cls_hidden, feat, conf_t)
            except Exception:
                pass

        self._update_source_weights(reward)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self._steps_since_update += 1
        self._maybe_update()

    def _maybe_update(self):
        if len(self.replay_buffer) >= ROLLOUT_SIZE or (
                self._steps_since_update >= MAX_STEPS_BEFORE_FLUSH
                and len(self.replay_buffer) >= MIN_STEPS_PER_AGENT * 2):
            self._ppo_update()

    # ---------------------------------------------------------- MAPPO
    def _ppo_update(self):
        """联盟式 MAPPO：每个有足够样本的 Agent 各自 PPO-Clip 更新。"""
        by_role: Dict[str, List[Transition]] = {}
        for tr in self.replay_buffer:
            by_role.setdefault(tr.agent, []).append(tr)

        logs: Dict[str, Dict[str, float]] = {}
        for role, agent in self._agents.items():
            trs = by_role.get(role) or []
            if len(trs) < MIN_STEPS_PER_AGENT:
                continue
            logs[role] = self._update_agent(agent, trs)

        self.replay_buffer.clear()
        self._steps_since_update = 0
        self.training_step += 1
        if self.training_step % LEAGUE_INTERVAL == 0:
            self._league_step()
        return logs

    def _update_agent(self, agent: PolicyAgent, trs: List[Transition]) -> Dict[str, float]:
        hp = agent.hp
        obs = torch.as_tensor(np.stack([t.obs for t in trs]), dtype=torch.float32)
        ctx = torch.as_tensor(np.stack([t.ctx for t in trs]), dtype=torch.float32)
        nobs = torch.as_tensor(np.stack([t.next_obs for t in trs]), dtype=torch.float32)
        nctx = torch.as_tensor(np.stack([t.next_ctx for t in trs]), dtype=torch.float32)
        rew = torch.as_tensor([t.reward for t in trs], dtype=torch.float32)
        done = torch.as_tensor([1.0 if t.done else 0.0 for t in trs], dtype=torch.float32)
        old_logp = torch.as_tensor([t.logprob for t in trs], dtype=torch.float32)
        old_v = torch.as_tensor([t.value for t in trs], dtype=torch.float32)
        actions = agent.stack_actions(trs)

        obs_ctx = torch.cat([obs, ctx], dim=-1)
        next_obs_ctx = torch.cat([nobs, nctx], dim=-1)

        # --- GAE（悲观 min 价值）+ 反事实优势 ---
        with torch.no_grad():
            v, v_std = agent.critic(obs_ctx, "min")
            nv, _ = agent.critic(next_obs_ctx, "min")
            nv = nv * (1.0 - done)
            delta = rew + hp.gamma * nv - v
            adv = torch.zeros_like(rew)
            running = torch.zeros((), dtype=torch.float32)
            for i in range(len(rew) - 1, -1, -1):
                running = delta[i] + hp.gamma * hp.gae_lambda * (1.0 - done[i]) * running
                adv[i] = running
            ret = adv + v
            cf = agent.counterfactual_advantage(obs, actions)

        adv_n = _normalize(adv)
        cf_n = _normalize(cf)
        A = (1.0 - hp.cf_weight) * adv_n + hp.cf_weight * cf_n

        # --- 策略 PPO-Clip（含熵奖励 + KL 早停）---
        bsz = len(trs)
        approx_kl = 0.0
        epochs_run = 0
        for epoch in range(hp.ppo_epochs):
            perm = torch.randperm(bsz)
            for start in range(0, bsz, hp.minibatch):
                mb = perm[start:start + hp.minibatch]
                logp, ent = agent._logp_entropy(agent.policy_outputs(obs[mb]), actions[mb])
                ratio = (logp - old_logp[mb]).exp()
                surr1 = ratio * A[mb]
                surr2 = ratio.clamp(1.0 - hp.clip, 1.0 + hp.clip) * A[mb]
                pg_loss = -torch.min(surr1, surr2).mean()
                loss = pg_loss - hp.entropy_coef * ent.mean()

                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(),
                                               hp.max_grad_norm)
                agent.optimizer.step()

                approx_kl = float((old_logp[mb] - logp).mean().item())
                epochs_run = epoch + 1
            if approx_kl > 2.0 * hp.kl_target:
                break

        # --- 集成 Critic：bagging + 价值裁剪（悲观目标）---
        agent.critic_optimizer.zero_grad()
        for i, member in enumerate(agent.critic.members):
            idx = torch.randint(0, bsz, (bsz,))
            pred = member(obs_ctx[idx]).squeeze(-1)
            v_clip = old_v[idx] + (pred - old_v[idx]).clamp(-hp.clip, hp.clip)
            l1 = F.smooth_l1_loss(pred, ret[idx], reduction="none")
            l2 = F.smooth_l1_loss(v_clip, ret[idx], reduction="none")
            (hp.value_coef * torch.max(l1, l2).mean()).backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), hp.max_grad_norm)
        agent.critic_optimizer.step()

        return {
            "steps": float(bsz),
            "adv_mean": float(adv.mean()),
            "ret_mean": float(ret.mean()),
            "kl": approx_kl,
            "entropy": float(ent.mean()),
            "critic_std": float(v_std.mean()),
            "epochs": float(epochs_run),
        }

    # ---------------------------------------------------------- 联盟
    def _league_step(self):
        """联盟评估 → 晋升/淘汰 → 集成蒸馏（防过拟合 + 保持多样性）。"""
        report = self._league_tournament()
        if report.get("promote"):
            player = self.league.promote(
                self.intent_agent.q_network, report["accuracy"],
                self.training_step, INTENT_HP.name)
            if player is not None:
                print(f"  [MARL 联盟] 晋升 #{player.pid} acc={report['accuracy']:.3f} "
                      f"(联盟 {len(self.league)}，h2h={report['h2h']:.2f})")
        self._distill_ensemble()

    @torch.no_grad()
    def _league_tournament(self) -> Dict[str, Any]:
        """用模板集评估当前策略 vs 联盟各成员（head-to-head）。"""
        cases: List[Tuple[str, IntentType]] = []
        for intent, templates in TASK_TEMPLATES.items():
            for tpl in templates[:5]:
                f = random.choice(SAMPLE_FILES)
                g = random.choice([x for x in SAMPLE_FILES if x != f])
                cases.append((tpl.format(f=f, g=g, c=SAMPLE_CONTENT), intent))
        if not cases:
            return {"accuracy": 0.0, "h2h": 0.0, "promote": False}

        stub = AgentState()
        stub.increment_round()
        cur_ok = 0
        per_player: Dict[int, List[bool]] = {p.pid: [] for p in self.league.players}
        opp_ok: Dict[int, int] = {p.pid: 0 for p in self.league.players}

        self.intent_agent.q_network.eval()
        for text, expected in cases:
            obs = self._obs("intent", text, stub)
            x = torch.FloatTensor(obs).unsqueeze(0)
            probs = F.softmax(self.intent_agent.q_network(x), dim=-1).squeeze(0).numpy()
            cur = self.intent_agent.action_map[int(probs.argmax())]
            hit = (cur == expected)
            cur_ok += int(hit)
            for p in self.league.players:
                p_probs = F.softmax(p.net(x), dim=-1).squeeze(0).numpy()
                p_hit = (self.intent_agent.action_map[int(p_probs.argmax())] == expected)
                per_player[p.pid].append(hit and not p_hit)
                opp_ok[p.pid] += int(p_hit)
        self.intent_agent.q_network.train()

        acc = cur_ok / len(cases)
        if not self.league.players:
            return {"accuracy": acc, "h2h": 0.0, "promote": True}

        h2h_scores = []
        for p in self.league.players:
            n = len(cases)
            wins = sum(per_player[p.pid])
            losses = opp_ok[p.pid] - wins
            h2h_scores.append((wins + 0.5 * (n - wins - losses)) / n)
        best_h2h = max(h2h_scores)
        best_opp_acc = max(opp_ok[p.pid] / len(cases) for p in self.league.players)
        promote = (best_h2h >= PROMOTE_THRESHOLD) or (acc >= best_opp_acc + 0.03)
        return {"accuracy": acc, "h2h": best_h2h, "promote": promote}

    def _distill_ensemble(self, steps: int = 15, lr: float = 7.5e-5):
        """集成策略蒸馏：把联盟集成的软标签吸收进当前单个网络（鲁棒性）。"""
        if not len(self.league) or len(self._distill_bank) < 16:
            return
        sample = random.sample(list(self._distill_bank),
                               min(96, len(self._distill_bank)))
        obs_t = torch.FloatTensor(np.stack(sample))
        targets = []
        self.intent_agent.q_network.eval()
        with torch.no_grad():
            for obs in sample:
                _, lg, _ = self._policy_probs(obs, include_league=True)
                if lg is None:
                    targets.append(F.softmax(self.intent_agent.q_network(
                        torch.FloatTensor(obs).unsqueeze(0)), dim=-1).squeeze(0))
                else:
                    cur = F.softmax(self.intent_agent.q_network(
                        torch.FloatTensor(obs).unsqueeze(0)), dim=-1).squeeze(0)
                    lg_t = torch.FloatTensor(lg)
                    targets.append(0.5 * cur + 0.5 * lg_t)
        self.intent_agent.q_network.train()
        if not targets:
            return
        target_t = torch.stack(targets)
        opt = optim.Adam(self.intent_agent.q_network.parameters(), lr=lr)
        for _ in range(steps):
            logits = self.intent_agent.q_network(obs_t)
            logp = F.log_softmax(logits, dim=-1)
            loss = -(target_t.detach() * logp).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.intent_agent.q_network.parameters(), 0.5)
            opt.step()

    # ---------------------------------------------------------- 快速训练
    def train_episode(self, num_episodes: int = 10):
        """模板自博弈预热：收集 rollout → MAPPO 更新 → 联盟评估（含对手扰动）。"""
        print(f"\n  [MARL Training] 联盟式预热 {num_episodes} 回合...")
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            intents = list(TASK_TEMPLATES.keys())
            random.shuffle(intents)
            for intent in intents:
                for template in TASK_TEMPLATES[intent]:
                    f = random.choice(SAMPLE_FILES)
                    g = random.choice([x for x in SAMPLE_FILES if x != f])
                    query = template.format(f=f, g=g, c=SAMPLE_CONTENT)

                    # 对手扰动：意图保持的查询改写，持续改变训练分布
                    arm = "identity"
                    if random.random() < OPP_PROB:
                        arm = self.query_opponent.choose()
                        query = self.query_opponent.apply(query, arm)

                    state = AgentState()
                    state.increment_round()
                    state.add_dialog(query)
                    predicted, _, confidence, _ = self.process(query, state)

                    reward = 5.0 if predicted == intent else -2.0
                    total_reward += reward
                    self.learn(reward, state, done=True)
                    self.query_opponent.update(arm, reward)

            self.reward_history.append(total_reward)
            print(f"    Episode {episode}/{num_episodes}: Reward={total_reward:.1f}, "
                  f"ε={self.epsilon:.4f}, 更新={self.training_step}, 联盟={len(self.league)}")

        if self.replay_buffer:
            self._ppo_update()
        self.save()
        print("  [MARL Training] 完成并保存模型")

    # ---------------------------------------------------------- 奖励/标签
    def compute_reward(self, outcome: str, state: AgentState = None) -> float:
        """与单 Agent 共用同一份奖励表与轮次衰减折算（避免数值漂移）。"""
        return round_reward(outcome, AgentRL.REWARD_MAP, state)

    def get_intent_label(self, intent: IntentType) -> str:
        return SHORT_LABELS.get(intent, intent.value)


# ================================================================
#  10. 语义特征提取器（与单 Agent 同构，观测基在 ObservationEncoder 处分化）
# ================================================================
class _SemanticFeatureExtractor:
    """字符 n-gram 哈希 + 固定随机投影 → 256 维稠密向量。"""

    HASH_DIM = 2048
    PROJ_DIM = SEM_DIM

    def __init__(self):
        import hashlib
        self._hashlib = hashlib
        rng = np.random.RandomState(42)
        self._proj = rng.randn(self.HASH_DIM, self.PROJ_DIM).astype(np.float32) * (
            1.0 / np.sqrt(self.HASH_DIM))

    def _char_ngrams(self, text: str) -> List[str]:
        text = text.lower().strip()
        ngrams = []
        for n in (1, 2, 3):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i + n])
        return ngrams

    def _hash_to_bucket(self, s: str) -> int:
        h = int(self._hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        return h % self.HASH_DIM

    def extract(self, query: str) -> np.ndarray:
        ngrams = self._char_ngrams(query)
        if not ngrams:
            return np.zeros(self.PROJ_DIM, dtype=np.float32)
        vec = np.zeros(self.HASH_DIM, dtype=np.float32)
        for ng in ngrams:
            vec[self._hash_to_bucket(ng)] += 1.0
        tf = np.log1p(vec)
        proj = tf @ self._proj
        norm = np.linalg.norm(proj)
        if norm > 0:
            proj /= norm
        return proj.astype(np.float32)

    @property
    def dim(self) -> int:
        return self.PROJ_DIM


# ================================================================
#  11. 工厂
# ================================================================
def create_marl_agent() -> MetaAgent:
    """完整 MARL Agent（启用单 Agent 交叉路，构成 6 路投票）。"""
    return MetaAgent(single_cross_vote=True)


def create_marl_voter() -> MetaAgent:
    """第 6 路投票器（只用策略委员会投票，不建单 Agent 交叉路 → 避免回环构造）。"""
    return MetaAgent(single_cross_vote=False)
