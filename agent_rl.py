"""Intent Classification Agent v2 — 语义特征 + 分类蒸馏 + Double DQN。

架构:
  1. SemanticFeatureExtractor — 中文字符 n-gram 哈希 → 256 维稠密向量
  2. DuelingQNetwork — Dueling + Double DQN 结构
  3. PrioritizedReplayBuffer — 按 TD-error 比例采样
  4. ClassificationHead — 交叉熵分类 + LLM 知识蒸馏
  5. ConfidenceArbiter — LLM / 规则 / Q-网络 置信度仲裁
  6. 第 6 路：MARL 策略委员会（get_marl_voter() 惰性单例，恒定开启）

六路投票（classify_intent）:
  llm / qnet / cls / rule / retrieval / marl
  marl 路由独立的 MAPPO 训练的 marl_agent.MetaAgent 提供（single_cross_vote=False，
  不建单 Agent 交叉路 → 与 MetaAgent 的第 6 路 single 互不回环）。
"""

import csv
import hashlib
import math
import os
import random
import re
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, NamedTuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent_state import IntentType, QuerySubIntent, RoutingPath, AgentState, QUERY_SUBINTENT_LABELS
from intent_shared import (
    SemanticFeatureExtractor,
    aggregate_votes,
    apply_readonly_guard,
    detect_readonly_guard,
    intent_label,
    query_routing_path as _shared_query_routing_path,
    route_intent as _shared_route_intent,
    semantic_conflict_guard as _shared_conflict_guard,
    train_reward as _shared_train_reward,
    train_pool as _shared_train_pool,
    eval_pool as _shared_eval_pool,
    vote_confidence as _shared_vote_confidence,
    # 关键词表：唯一定义在 intent_shared，_safe_fallback 不再各写一份
    SEARCH_KEYWORDS,
    DELETE_KEYWORDS,
    DEEP_ANALYSIS_KEYWORDS,
    ADD_FILE_KEYWORDS,
    MODIFY_FILE_KEYWORDS,
    REFACTOR_KEYWORDS,
    FILENAME_QUERY_KEYWORDS,
    WRITE_KEYWORDS,
    ANALYSIS_GATE_KEYWORDS,
    FORBIDDEN_MODIFY_KEYWORDS,
    SCAN_KEYWORDS,
    REPORT_KEYWORDS,
    CONFIDENCE_THRESHOLD as SHARED_CONFIDENCE_THRESHOLD,
    ENGRAM_MIN_MARGIN as SHARED_ENGRAM_MIN_MARGIN,
    TASK_TEMPLATES as SHARED_TASK_TEMPLATES,
    SAMPLE_FILES as SHARED_SAMPLE_FILES,
    SAMPLE_CONTENT as SHARED_SAMPLE_CONTENT,
)
from atomic_io import read_json as read_json_file
from query_rewriter import QueryRewriter
from llm_intent import LLMIntentClassifier
from intent_retriever import get_intent_retriever
from transformer_arch import (
    TransformerDuelingQNetworkWithNSP, TransformerClassificationHead,
    EngramConditionalMemory,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "agent_rl.pth")
MODEL_VERSION_DIR = os.path.join(MODEL_DIR, "versions")

MARL_VOTE_WEIGHT = 0.62        # 有 LLM 时第 6 路权重
MARL_VOTE_WEIGHT_NO_LLM = 0.78  # 无 LLM 时提升为强兜底

_MARL_VOTER = None
_MARL_VOTER_FAILED = False
# 可重入：第 6 路构造过程中若再次回到本函数（回环），必须能重新拿到锁
# 才有机会查下面的 _MARL_VOTER_BUILDING 并安全退出；普通 Lock 会直接死锁。
_MARL_VOTER_LOCK = threading.RLock()
#: 构造进行中标记：重入时说明出现了 agent_rl → marl_agent → agent_rl 的
#: 构造回环，此时返回 None（放弃第 6 路）而不是递归构造下去。
_MARL_VOTER_BUILDING = False


def get_marl_voter():
    """第 6 路投票器（MARL 策略委员会）惰性线程安全单例。

    返回 marl_agent.MetaAgent(single_cross_vote=False) 或 None（初始化失败，
    缓存失败结果避免反复构造）。调用方必须 try/except 包裹 vote() 调用。

    防回环：``create_marl_voter`` 必须是 ``single_cross_vote=False`` 的
    MetaAgent（它内部不会再造 AgentRL）；万一将来有人改破这条不变式，
    重入检测会在这里返回 None，把构造回环降级成「第 6 路不可用」，
    而不是无限递归 / 死锁。
    """
    global _MARL_VOTER, _MARL_VOTER_FAILED, _MARL_VOTER_BUILDING
    with _MARL_VOTER_LOCK:
        if _MARL_VOTER_BUILDING:
            # 构造回环：外层正在建第 6 路，本次调用放弃本路
            print("  [MARL] 检测到第 6 路构造回环，本次返回 None（防递归）。")
            return None
        if _MARL_VOTER is not None:
            return _MARL_VOTER
        if _MARL_VOTER_FAILED:
            return None
        _MARL_VOTER_BUILDING = True
        try:
            from marl_agent import create_marl_voter
            _MARL_VOTER = create_marl_voter()
        except Exception as e:
            _MARL_VOTER_FAILED = True
            print(f"  [MARL] 第 6 路投票器初始化失败({e})，该路不可用。")
        finally:
            _MARL_VOTER_BUILDING = False
        return _MARL_VOTER


# ================================================================
#  1. 语义特征提取器 — 见 intent_shared.SemanticFeatureExtractor（唯一副本）
# ================================================================

# ================================================================
#  2. Dueling + Double DQN
# ================================================================
class DuelingQNetwork(nn.Module):
    """Dueling DQN: V(s) + A(s,a) → Q(s,a)。"""

    def __init__(self, state_dim: int = 268, hidden_dim: int = 192, num_actions: int = 7):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        return val + adv - adv.mean(dim=-1, keepdim=True)


# ================================================================
#  3. 分类头 — 交叉熵 + 蒸馏
# ================================================================
class ClassificationHead(nn.Module):
    """轻量分类头：语义特征 → 7 类意图概率。用于有监督预训练 + 蒸馏。"""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ================================================================
#  4. Prioritized Replay Buffer
# ================================================================
class SumTree:
    """二叉树求和，O(log n) 按优先级采样。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_pos = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        # idx<=0 时 parent 会变成 -1（Python 负索引会写穿 tree[-1]，且永不终止）
        if parent < 0:
            return
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """按 TD-error 比例采样的经验回放。

    稳定性措施（对应路线图「软更新 + PER 稳定在线策略」）：
      - ``td_clip``：更新优先级前裁剪 TD-error，单个离群样本无法垄断采样
      - ``min_priority``：新样本优先级下限，避免 priority 被少数大样本压扁后
        新经验长期采不到（新经验饿死）
      - ``beta``：IS 权重指数从 0.4 退火到 1.0（每采样一次 +beta_inc）
    """

    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4,
                 td_clip: float = 10.0, beta_inc: float = 0.0001):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.td_clip = td_clip
        self.max_priority = 1.0
        self.min_priority = 0.05 ** alpha
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, state, action, reward, next_state, done):
        priority = max(self.max_priority ** self.alpha, self.min_priority)
        self.tree.add(priority, (state, action, reward, next_state, done))
        self._size = min(self._size + 1, self.tree.capacity)

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        if self.tree.total <= 0:
            return [], np.array([]), np.array([])
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total / batch_size

        self.beta = min(1.0, self.beta + self.beta_inc)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        if not batch:
            return [], np.array([]), np.array([])

        probs = np.array(priorities, dtype=np.float64)
        probs = probs / (probs.sum() + 1e-8)
        weights = (self._size * probs + 1e-8) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)

        return batch, np.array(indices, dtype=np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        # NaN/Inf 会把整棵优先级树算废：先清洗，再裁剪到 td_clip
        tds = np.nan_to_num(np.asarray(td_errors, dtype=np.float64),
                            nan=0.0, posinf=self.td_clip, neginf=-self.td_clip)
        tds = np.clip(tds, -self.td_clip, self.td_clip)
        for idx, td in zip(indices, tds):
            priority = (abs(td) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority ** (1.0 / self.alpha))
            self.tree.update(int(idx), priority)

    # ---------------------------------------------------- 持久化（buffer 落盘）
    MAX_PERSIST_SAMPLES = 20000   # 落盘样本上限，避免 checkpoint 膨胀

    def _chronological_indices(self) -> List[int]:
        """按写入先后返回有效槽位下标（旧 → 新）。

        环形缓冲：未写满时槽位 0..n-1 即顺序；写满后最旧的是 write_pos。
        """
        cap, n, wp = self.tree.capacity, self.tree.n_entries, self.tree.write_pos
        if n < cap:
            return list(range(n))
        return [(wp + k) % cap for k in range(cap)]

    def to_arrays(self, max_samples: int = None) -> Optional[Dict[str, "np.ndarray"]]:
        """导出为 numpy 数组（超出上限时只保留最新的）。buffer 为空返回 None。"""
        if self._size <= 0:
            return None
        limit = self.MAX_PERSIST_SAMPLES if max_samples is None else max_samples
        indices = self._chronological_indices()[-limit:]

        states, actions, rewards, next_states, dones, priorities = [], [], [], [], [], []
        for slot in indices:
            item = self.tree.data[slot]
            if item is None:
                continue
            s, a, r, ns, d = item
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            priorities.append(self.tree.tree[slot + self.tree.capacity - 1])

        if not states:
            return None
        return {
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "priorities": np.asarray(priorities, dtype=np.float64),
            "alpha": np.asarray(self.alpha, dtype=np.float64),
            "beta": np.asarray(self.beta, dtype=np.float64),
            "max_priority": np.asarray(self.max_priority, dtype=np.float64),
        }

    def from_arrays(self, arrays: Dict[str, "np.ndarray"]) -> int:
        """从 ``to_arrays`` 的输出恢复 buffer，返回恢复的样本数。"""
        states = np.asarray(arrays["states"])
        if states.size == 0:
            return 0
        actions = np.asarray(arrays["actions"])
        rewards = np.asarray(arrays["rewards"])
        next_states = np.asarray(arrays["next_states"])
        dones = np.asarray(arrays["dones"])
        priorities = np.asarray(arrays.get(
            "priorities", np.full(len(states), self.max_priority ** self.alpha)
        ), dtype=np.float64)

        # 清空再按序回填（顺序即旧→新，保证 write_pos 指向正确位置）
        self.tree.tree[:] = 0.0
        self.tree.data = [None] * self.tree.capacity
        self.tree.write_pos = 0
        self.tree.n_entries = 0
        self._size = 0

        for i in range(len(states)):
            self.tree.add(
                float(priorities[i]),
                (states[i], int(actions[i]), float(rewards[i]),
                 next_states[i], bool(dones[i])),
            )
        self._size = self.tree.n_entries
        self.alpha = float(arrays.get("alpha", self.alpha))
        self.beta = float(arrays.get("beta", self.beta))
        self.max_priority = float(arrays.get("max_priority", self.max_priority))
        return self._size


# ================================================================
#  4.5 澄清闭环：轮次衰减奖励 + 置信度期望决策
# ================================================================
# 记号（k = 澄清深度，k=1 表示首轮直接回答；q=询问奖励，a=答对奖励）：
#   路径总奖励 P_k = Σ_{i<k} q_i + a_k
# 递推定义 a_1 = base，a_{j+1} = a_j - q_j - eps_j，其中
#   eps_1 = -CLARIFY_BONUS (<0) → P2 = P1 + BONUS > P1      约束①
#   eps_2 =  0                   → P3 = P2                   约束②
#   eps_j = +CLARIFY_EXTRA(j≥3) → P4 起严格下降且 < P1        约束③
# 答错/超时保持平表 R_WRONG，不随轮次衰减。
CLARIFY_Q_BASE = 1.5       # q_1：首轮澄清兑现的询问奖励
CLARIFY_Q_DECAY = 0.8      # q_k = q_1 * 0.8^(k-1)，询问奖励逐轮衰减
CLARIFY_BONUS = 1.0        # 首次澄清的信息增益（P2 = P1 + 1.0）
CLARIFY_EXTRA = 1.5        # 第 3 轮起每多问一轮的额外惩罚
R_WRONG = -4.0             # 答错基准（*_failure）
MAX_CLARIFY_DEPTH = 3      # 澄清深度封顶，超过后强制回答
EV_BETA = 0.4              # 对"下一轮才拿到答案"的折扣（用户打断成本）
EV_ETA = 0.8               # 澄清对剩余不确定性的解析率
EV_NOMINAL_BASE = 6.0      # 计算期望阈值时的名义答对奖励

# 答对类奖励键按轮次衰减；失败/超时类保持平表
ANSWER_OUTCOMES = frozenset({
    "add_success", "delete_success", "modify_success", "refactor_success",
    "query_success", "deep_search_success", "evidence_sufficient",
})


def ask_reward(k: int) -> float:
    """第 k 轮澄清的询问奖励 q_k（指数衰减）。"""
    return CLARIFY_Q_BASE * (CLARIFY_Q_DECAY ** (k - 1))


def _extra_decay(j: int) -> float:
    """第 j → j+1 轮之间的额外衰减项 eps_j。"""
    if j <= 1:
        return -CLARIFY_BONUS
    if j == 2:
        return 0.0
    return CLARIFY_EXTRA


def answer_reward(base: float, k: int) -> float:
    """深度 k 处答对的奖励 a_k：a_1 = base，a_{j+1} = a_j - q_j - eps_j。"""
    a = float(base)
    for j in range(1, k):
        a -= ask_reward(j)
        a -= _extra_decay(j)
    return max(a, R_WRONG + 0.5)


def confidence_threshold(k: int) -> float:
    """期望决策阈值 p*：conf < p* 牺牲奖励继续询问，conf >= p* 冒着答错风险回答。

      EV_answer(p) = R_WRONG + p * (a_k - R_WRONG)
      EV_ask(p)    = q_k + BETA * [R_WRONG + p' * (a_{k+1} - R_WRONG)]
      p' = ETA + (1 - ETA) * p
    令 EV_ask = EV_answer 解出闭式 p*。
    """
    a_k = answer_reward(EV_NOMINAL_BASE, k)
    a_next = answer_reward(EV_NOMINAL_BASE, k + 1)
    q_k = ask_reward(k)
    num = q_k + EV_BETA * (R_WRONG + EV_ETA * (a_next - R_WRONG)) - R_WRONG
    den = (a_k - R_WRONG) - EV_BETA * (1.0 - EV_ETA) * (a_next - R_WRONG)
    if den <= 1e-9:
        return 1.0
    return min(0.95, max(0.30, num / den))


def should_clarify(confidence: float, state: Optional[AgentState]) -> bool:
    """置信度期望决策：是否牺牲奖励继续询问而非直接回答。"""
    depth = state.clarification_rounds if state is not None else 0
    if depth >= MAX_CLARIFY_DEPTH:
        return False                       # 深度封顶，强制回答
    return confidence < confidence_threshold(depth + 1)


def round_reward(outcome: str, reward_map: Dict[str, float],
                 state: Optional[AgentState]) -> float:
    """按澄清深度折算奖励：询问 q_k 衰减、答对 a_k 衰减、答错固定。"""
    base = reward_map.get(outcome, 0.0)
    if state is None:
        return base
    k = state.clarification_rounds + 1
    if outcome == "clarification_resolved":
        return round(ask_reward(k), 4)
    if outcome in ANSWER_OUTCOMES:
        return round(answer_reward(base, k), 4)
    return base


# ================================================================
#  5. 主 Agent
# ================================================================
class AgentRL:
    """意图分类 Agent：分类头 + Dueling Double DQN + 蒸馏。

    特征:
      - 256 维语义特征（字符 n-gram 哈希） + 12 维状态向量 = 268 维输入
      - 分类头：交叉熵有监督训练（模板 + 蒸馏标签）
      - DQN：Double DQN + Dueling + Prioritized Replay
      - 置信度仲裁：LLM / 规则 / Q网络 / 分类头 / 检索 / MARL 六方投票
    """

    INTENT_ORDER = [
        IntentType.ADD_FILE, IntentType.DELETE_FILE, IntentType.MODIFY_FILE,
        IntentType.REFACTOR, IntentType.QUERY, IntentType.DEEP_SEARCH, IntentType.VAGUE,
        IntentType.FILE_ANALYSIS, IntentType.FILE_SCAN, IntentType.REPORT_GENERATE,
        IntentType.FILENAME_QUERY,
    ]
    FEATURE_DIM = 268
    CONFIDENCE_THRESHOLD = SHARED_CONFIDENCE_THRESHOLD  # 与 marl_agent 同源，见 intent_shared

    # Engram 外部记忆的补充权重（详见 classify_intent 中的调用处）：
    # 记忆只做补充，不构成独立决策方，改判必须比主体判断高出 ENGRAM_MIN_MARGIN。
    ENGRAM_WEIGHT = 0.25
    ENGRAM_MIN_MARGIN = SHARED_ENGRAM_MIN_MARGIN

    # 奖励默认表：marl_agent.MetaAgent.compute_reward 直接引用本表，
    # 任何新增奖励键只在此处添加一处。
    # config/app.yaml 的 intent_llm.reward 会在 AgentRL.__init__ 中按键合并覆盖本表
    #（键名先经 rl_config.REWARD_KEY_ALIASES 归一到本表规范键）。
    # 注意：带 state 调用时，答对键与 clarification_resolved 会被
    # round_reward() 按澄清深度折算（a_k / q_k 衰减），本表为 k=1 的基准值。
    REWARD_MAP: Dict[str, float] = {
        "add_success": 5.0, "delete_success": 5.0, "modify_success": 5.0,
        "refactor_success": 7.0, "query_success": 5.0, "deep_search_success": 8.0,
        "add_failure": -4.0, "delete_failure": -4.0, "modify_failure": -4.0,
        "refactor_failure": -4.0, "insufficient_evidence": -2.0,
        "clarification_resolved": 7.0,
        "timeout": -3.0, "evidence_sufficient": 8.0,
    }

    # 训练模板 / 演示样本：唯一副本在 intent_shared，两个 agent 共用一份，
    # 避免复制后各自漂移（marl 侧原有 2 条模板与单体不一致）。
    TASK_TEMPLATES = SHARED_TASK_TEMPLATES
    SAMPLE_FILES = SHARED_SAMPLE_FILES
    SAMPLE_CONTENT = SHARED_SAMPLE_CONTENT

    # --- 蒸馏用真实测试样本（不在模板中） ---
    EVAL_REAL_SAMPLES: Dict[str, str] = {
        "add_file": "帮我新建一个 readme.md 项目说明文档",
        "delete_file": "把那个旧的 backup.py 删了",
        "modify_file": "更新一下 config.yaml 里的超时时间",
        "refactor": "把 src 下面的文件按功能分一下类",
        "query": "这个项目用了什么框架",
        "deep_search": "分析一下这个项目的架构和性能瓶颈",
        "vague": "你好呀",
        "vague_2": "今天过得怎么样",
        "vague_3": "帮我",
        "file_analysis": "只读分析项目代码质量，不修改任何文件",
        "file_scan": "扫描项目中的所有Python文件",
        "report_generate": "生成代码审查报告，输出Markdown格式",
        "filename_query": "为我找到最近添加的文件名",
        "filename_query_2": "最近修改的文件有哪些",
        "filename_query_3": "查找最近创建的txt文件",
    }

    def __init__(self, model_path: str = None, marl_vote: bool = True):
        self.marl_vote = marl_vote  # 第 6 路开关（默认开启；MetaAgent 交叉路关闭）
        self.query_rewriter = QueryRewriter()
        self.state_dim = self.FEATURE_DIM
        self.num_actions = len(self.INTENT_ORDER)
        self.action_map: Dict[int, IntentType] = {i: it for i, it in enumerate(self.INTENT_ORDER)}
        self.reverse_map: Dict[IntentType, int] = {it: i for i, it in self.action_map.items()}

        self.feature_extractor = SemanticFeatureExtractor()
        self.classifier = TransformerClassificationHead(
            self.state_dim, 128, self.num_actions, num_loops=4
        )
        self.q_network = TransformerDuelingQNetworkWithNSP(
            self.state_dim, 128, num_actions=self.num_actions,
            num_experts=8, num_loops=6
        )
        self.target_network = TransformerDuelingQNetworkWithNSP(
            self.state_dim, 128, num_actions=self.num_actions,
            num_experts=8, num_loops=6
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        # NSP 辅助任务超参
        self.nsp_lambda = 0.1  # NSP 损失权重

        # Engram Conditional Memory — 解决幻觉问题
        self.engram = EngramConditionalMemory(d_model=128, input_dim=self.state_dim)

        all_params = list(self.q_network.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.Adam(all_params, lr=3e-4)
        self.replay_buffer = PrioritizedReplayBuffer(50000)

        self.epsilon = 0.3
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.gamma = 0.99  # 默认值；config/app.yaml rl.gamma 在 _apply_app_config() 中覆盖
        # 软更新调度：训练早期 target 需要快速跟上网络（tau_init 大），
        # 后期逐渐收紧到 tau_base 让 target 平滑逼近，避免在线策略抖动。
        self.tau = 0.005
        self.tau_init = 0.05
        self.tau_base = 0.005
        self.tau_anneal_steps = 10000
        self.batch_size = 64
        self.training_step = 0

        self.intent_history: List[Tuple[int, float]] = []
        self.reward_history: List[float] = []
        self.last_sources: List[str] = []
        self.trained = False

        self.last_features: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

        self.llm_classifier = LLMIntentClassifier()
        self.last_llm_target: Optional[str] = None
        self.last_llm_sub_intent: Optional[str] = None
        self.last_llm_slots: Dict[str, str] = {}

        # 检索式意图识别：标注语料稠密向量召回，作为第四路投票信号
        self.intent_retriever = get_intent_retriever()

        # config/app.yaml：intent_llm.reward → REWARD_MAP，rl.gamma → 折扣因子
        self._apply_app_config()

        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model_version: Optional[str] = None
        self.loaded_from_disk = False
        self._auto_load()

    # ---------------------------------------------------------- 配置
    def _apply_app_config(self):
        """读 config/app.yaml：把 intent_llm.reward 合并进 REWARD_MAP，并用 rl.gamma 覆盖折扣因子。

        REWARD_MAP 是类属性（marl_agent.MetaAgent.compute_reward 直接引用
        AgentRL.REWARD_MAP），故此处 update 对所有实例与 MetaAgent 同时生效。
        奖励键在 rl_config.REWARD_KEY_ALIASES 中先归一到代码实际使用的
        query_success / add_failure 等规范键。
        """
        from rl_config import get_rl_gamma, get_reward_overrides

        overrides = get_reward_overrides()
        if overrides:
            unknown = sorted(k for k in overrides if k not in self.REWARD_MAP)
            self.REWARD_MAP.update(overrides)
            extra = f"，新增键 {unknown}" if unknown else ""
            print(f"  [配置] config/app.yaml: 合并 {len(overrides)} 项奖励 → REWARD_MAP{extra}")
        self.gamma = get_rl_gamma()

    # ---------------------------------------------------------- 持久化
    def _auto_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            self.load(self.model_path)
            self.loaded_from_disk = True
            print(f"\n  [持久化] 已加载模型 {self.model_path} "
                  f"(ε={self.epsilon:.4f}, 步数={self.training_step}, 池={len(self.replay_buffer)})")
        except Exception as e:
            self.loaded_from_disk = False
            print(f"  [持久化] 加载失败({e})，重新初始化。")

    def save(self, path: str = None, save_replay: bool = True):
        """原子保存 checkpoint（模型权重 + 超参），可选一并持久化 replay buffer。

        replay buffer 走同目录 sidecar ``<path>.replay.npz``：
        把体积大的经验池和权重分开，主 checkpoint 保持轻量、可单独热加载。
        """
        from atomic_io import atomic_torch_save

        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        replay_saved = False
        if save_replay:
            replay_saved = self._save_replay(self.replay_path_for(path))

        payload = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "classifier": self.classifier.state_dict(),
            "engram": self.engram.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "nsp_lambda": self.nsp_lambda,
            "training_step": self.training_step,
            "trained": self.trained,
            # --- 软更新调度参数（跨进程保持退火进度一致） ---
            "tau_init": self.tau_init,
            "tau_base": self.tau_base,
            "tau_anneal_steps": self.tau_anneal_steps,
            # --- 版本与回放元数据 ---
            "model_version": self.model_version,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "replay_file": os.path.basename(self.replay_path_for(path)) if replay_saved else None,
            "replay_size": len(self.replay_buffer),
        }
        atomic_torch_save(path, payload)
        print(f"  [持久化] 已保存 → {path} (步数 {self.training_step}"
              + (f", buffer {len(self.replay_buffer)}" if replay_saved else "") + ")")

    def replay_path_for(self, model_path: str) -> str:
        return model_path + ".replay.npz"

    def _save_replay(self, replay_path: str) -> bool:
        """把经验池压成 npz sidecar（空 buffer 则删除旧 sidecar）。"""
        import tempfile

        arrays = self.replay_buffer.to_arrays()
        if arrays is None:
            try:
                if os.path.exists(replay_path):
                    os.unlink(replay_path)
            except OSError:
                pass
            return False

        directory = os.path.dirname(os.path.abspath(replay_path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(replay_path) + ".",
            suffix=".npz",          # 必须以 .npz 结尾，否则 np.savez 会再补后缀
            dir=directory,
        )
        os.close(fd)
        try:
            np.savez_compressed(tmp_path, **arrays)
            # Windows 上 FlushFileBuffers 要求写权限，只读句柄会 EBADF
            with open(tmp_path, "r+b") as fh:
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, replay_path)   # 同目录原子替换
            from atomic_io import fsync_dir
            fsync_dir(directory)
            return True
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load_replay(self, model_path: str, replay_file: str = None) -> int:
        """加载 replay sidecar；文件缺失/损坏时返回 0，不阻塞权重加载。"""
        candidates = []
        if replay_file:
            candidates.append(os.path.join(os.path.dirname(model_path), replay_file))
        candidates.append(self.replay_path_for(model_path))

        for candidate in candidates:
            if not candidate or not os.path.exists(candidate):
                continue
            try:
                with np.load(candidate) as data:
                    restored = self.replay_buffer.from_arrays(
                        {k: data[k] for k in data.files}
                    )
                if restored:
                    return restored
            except Exception as exc:
                print(f"  [持久化] replay sidecar 加载失败({exc})，跳过。")
        return 0

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "q_network" in ckpt:
            self.q_network.load_state_dict(ckpt["q_network"])
        if "target_network" in ckpt:
            self.target_network.load_state_dict(ckpt["target_network"])
        if "classifier" in ckpt:
            self.classifier.load_state_dict(ckpt["classifier"])
        if "engram" in ckpt:
            self.engram.load_state_dict(ckpt["engram"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                pass
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.nsp_lambda = ckpt.get("nsp_lambda", self.nsp_lambda)
        self.training_step = ckpt.get("training_step", self.training_step)
        self.trained = ckpt.get("trained", self.trained)
        self.tau_init = ckpt.get("tau_init", self.tau_init)
        self.tau_base = ckpt.get("tau_base", self.tau_base)
        self.tau_anneal_steps = ckpt.get("tau_anneal_steps", self.tau_anneal_steps)
        self.tau = self.tau_base
        self.model_version = ckpt.get("model_version")
        restored = self._load_replay(path, ckpt.get("replay_file"))
        if restored:
            print(f"  [持久化] 已恢复 replay buffer {restored} 条 → {path}.replay.npz")

    # ---------------------------------------------------------- 模型版本
    def save_version(self, name: str = "", note: str = "") -> str:
        """把当前模型（含 replay buffer）存成一个不可变版本快照。

        版本目录 ``models/versions/<version_id>/`` 内含 checkpoint、
        replay sidecar 与 meta.json；``models/versions/latest.json``
        指向最新版本，供回滚与灰度使用。
        """
        from atomic_io import atomic_write_json

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff-]+", "-", (name or "ckpt")).strip("-")
        version_id = f"v_{stamp}_{slug}" if slug else f"v_{stamp}"
        version_dir = os.path.join(MODEL_VERSION_DIR, version_id)
        os.makedirs(version_dir, exist_ok=True)

        ckpt_path = os.path.join(version_dir, "agent_rl.pth")
        self.save(ckpt_path, save_replay=True)
        self.model_version = version_id

        meta = {
            "version_id": version_id,
            "name": name,
            "note": note,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "replay_size": len(self.replay_buffer),
            "trained": self.trained,
        }
        atomic_write_json(os.path.join(version_dir, "meta.json"), meta)
        atomic_write_json(os.path.join(MODEL_VERSION_DIR, "latest.json"),
                          {"version_id": version_id, "updated_at": meta["created_at"]})
        print(f"  [版本] 已固化 → {version_dir}")
        return version_id

    def list_model_versions(self) -> List[Dict[str, Any]]:
        """列出所有模型版本（新→旧）。"""
        if not os.path.isdir(MODEL_VERSION_DIR):
            return []
        versions = []
        for entry in sorted(os.listdir(MODEL_VERSION_DIR), reverse=True):
            meta_path = os.path.join(MODEL_VERSION_DIR, entry, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    versions.append(read_json_file(meta_path))
                except Exception:
                    continue
        return versions

    def latest_model_version(self) -> Optional[str]:
        pointer = os.path.join(MODEL_VERSION_DIR, "latest.json")
        if not os.path.isfile(pointer):
            return None
        try:
            return read_json_file(pointer).get("version_id")
        except Exception:
            return None

    def load_version(self, version_id: str) -> bool:
        """回滚/切换到指定版本。返回是否加载成功。"""
        path = os.path.join(MODEL_VERSION_DIR, version_id, "agent_rl.pth")
        if not os.path.exists(path):
            return False
        self.load(path)
        self.model_version = version_id
        print(f"  [版本] 已切换到 {version_id}")
        return True

    # ---------------------------------------------------------- 特征
    def _extract_features(self, query: str, state: AgentState) -> np.ndarray:
        """268 维 = 语义(256) + 状态(12)。"""
        sem = self.feature_extractor.extract(query)
        state_vec = state.get_state_vector()
        return np.concatenate([sem, state_vec]).astype(np.float32)

    # -------------------------------------------------------- 分类预测
    @torch.no_grad()
    def _classify_probs(self, features: np.ndarray) -> np.ndarray:
        """分类头 softmax 概率。"""
        self.classifier.eval()
        x = torch.FloatTensor(features).unsqueeze(0)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
        self.classifier.train()
        return probs

    # -------------------------------------------------------- DQN 选择
    def select_action(self, features: np.ndarray, state: AgentState) -> Tuple[int, float]:
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
            confidence = 1.0 / self.num_actions
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(features).unsqueeze(0))
                action = q_values.argmax(dim=1).item()
                confidence = float(torch.softmax(q_values, dim=1).max().item())
        self.intent_history.append((action, confidence))
        self.last_features = features.copy()
        self.last_action = action
        return action, confidence

    # ---------------------------------------------------- 无副作用融合投票
    @torch.no_grad()
    def neural_vote(self, query: str, state: AgentState
                    ) -> Tuple[IntentType, float, Dict[str, Any]]:
        """cls + qnet 融合投票（只读，无副作用）。

        供 marl_agent.MetaAgent 的第 6 路 single 交叉路调用：
        不调 LLM、不触发第 6 路（不会构造 get_marl_voter → 无回环）、
        不做 ε 探索、不写 intent_history/last_features。
        返回 (意图, 融合置信度, 各路诊断信息)。
        """
        features = self._extract_features(query, state)
        self.q_network.eval()
        with torch.no_grad():
            q_logits = self.q_network(torch.FloatTensor(features).unsqueeze(0))
            q_probs = F.softmax(q_logits, dim=1).squeeze(0).numpy()
        self.q_network.train()
        cls_probs = self._classify_probs(features)
        blend = 0.55 * q_probs + 0.45 * cls_probs
        idx = int(blend.argmax())
        info = {
            "qnet_conf": float(q_probs[idx]),
            "cls_conf": float(cls_probs[idx]),
            "qnet_action": int(q_probs.argmax()),
            "cls_action": int(cls_probs.argmax()),
        }
        return self.action_map[idx], float(blend[idx]), info

    # --------------------------------------------------- 置信度仲裁
    def classify_intent(self, query: str, state: AgentState) -> Tuple[IntentType, float]:
        """六方仲裁：LLM → 规则 + Q网络 + 分类头 + 检索 + MARL → 保底。

        冲突时用置信度决定最终意图。
        规则在无 LLM 时权重提升，且对语义冲突有纠错能力。
        检索路（retrieval）从标注意图语料库召回相似例句，补齐 OOD/少样本泛化。
        MARL 路（marl）由独立 MAPPO 训练的策略委员会提供，恒定开启（不依赖
        LLM）；仅在投票器不可用或推理异常时跳过本路，其余五路不受影响。
        本次参与投票的来源记录于 self.last_sources。
        """
        self.last_llm_target = None
        self.last_llm_sub_intent = None
        self.last_llm_slots = {}
        candidates: List[Tuple[IntentType, float, str]] = []

        llm_result = self.llm_classifier.classify(query)
        llm_available = False
        if llm_result:
            llm_available = True
            intent_value, target, confidence, sub_intent, slots = llm_result
            try:
                intent = IntentType(intent_value)
                candidates.append((intent, confidence, "llm"))
                self.last_llm_target = target
                self.last_llm_sub_intent = sub_intent
                self.last_llm_slots = slots
            except ValueError:
                pass

        features = self._extract_features(query, state)
        q_action, q_conf = self.select_action(features, state)
        candidates.append((self.action_map[q_action], q_conf, "qnet"))

        cls_probs = self._classify_probs(features)
        cls_action = int(cls_probs.argmax())
        cls_conf = float(cls_probs[cls_action])
        candidates.append((self.action_map[cls_action], cls_conf, "cls"))

        top_op = self.query_rewriter.top_operation(query)
        if top_op != "vague":
            try:
                rule_intent = IntentType(top_op)
                rule_weight = 0.55 if llm_available else 0.85
                candidates.append((rule_intent, rule_weight, "rule"))
            except ValueError:
                pass

        # 第四路：检索式意图投票（相似例句召回；低相似度自动弃权）
        ret_intent, ret_conf, ret_evidence = self.intent_retriever.vote(query)
        if ret_intent is not None:
            # 有 LLM 时检索权重略降（LLM 语义更强），无 LLM 时提升为强兜底
            ret_weight = 0.50 if llm_available else 0.75
            candidates.append((ret_intent, ret_conf * ret_weight, "retrieval"))

        # 第六路：MARL 策略委员会（恒定开启；不依赖 LLM，异常时仅跳过本路）
        self._last_marl_info: Dict[str, Any] = {}
        if self.marl_vote:
            voter = get_marl_voter()
            if voter is not None:
                try:
                    m_intent, m_conf, m_info = voter.vote(query, state)
                except Exception as e:
                    m_intent, m_conf, m_info = None, 0.0, {"error": str(e)}
                self._last_marl_info = m_info if isinstance(m_info, dict) else {}
                if m_intent is not None:
                    marl_weight = MARL_VOTE_WEIGHT if llm_available else MARL_VOTE_WEIGHT_NO_LLM
                    candidates.append((m_intent, m_conf * marl_weight, "marl"))

        self.last_sources = list(dict.fromkeys(src for _, _, src in candidates))

        # 聚合：score = Σ各路得分 / n**0.75（= 平均质量 × n**0.25，共识增强）。
        # 旧实现 Σ/n 取均值 → 多一路同向共识反而降分；排序与置信度共用这一个数。
        votes = aggregate_votes(candidates)

        final_intent = max(votes, key=lambda k: votes[k]["score"])
        final_conf = _shared_vote_confidence(votes, final_intent)

        # Engram 外部记忆：**补充信号**，不是第七个独立决策方。
        #   - 只在投票置信度低于下限（模型吃不准）时才读取
        #   - 与主体判断一致 → 小幅抬升置信度（权重 ENGRAM_WEIGHT，封顶 1.0）
        #   - 与主体判断不一致 → 仅当记忆自身确信度高出 ENGRAM_MIN_MARGIN 才改判
        #   - 置信度足够高时完全不参与，绝不用外部记忆覆盖高置信度决策
        if final_conf < self.CONFIDENCE_THRESHOLD:
            eng_intent, eng_conf = self._engram_vote(features, final_conf)
            if eng_intent is not None:
                if eng_intent == final_intent:
                    final_conf = min(1.0, final_conf + self.ENGRAM_WEIGHT * eng_conf)
                elif eng_conf > final_conf + self.ENGRAM_MIN_MARGIN:
                    final_intent = eng_intent
                    final_conf = max(final_conf, eng_conf)
                if "engram" not in self.last_sources:
                    self.last_sources.append("engram")

            if final_conf < 0.4:
                # 极低置信度：强制VAGUE，要求澄清
                final_intent = IntentType.VAGUE
                final_conf = 0.3
            else:
                # 中等置信度：尝试安全回退
                safe = self._safe_fallback(query)
                if safe is not None:
                    final_intent = safe
                    final_conf = max(final_conf, 0.5)
                else:
                    final_intent = IntentType.VAGUE
                    final_conf = 0.35

        final_intent = self._semantic_conflict_guard(query, final_intent, votes)

        return final_intent, final_conf

    def _semantic_conflict_guard(self, query: str, proposed: IntentType,
                                  votes: Dict[IntentType, Dict] = None) -> IntentType:
        """语义冲突护栏 —— 委托 intent_shared（与 marl_agent 完全同一实现）。

        旧副本只拦 DELETE_FILE、且带一段 ``votes[IntentType.QUERY]["score"]``
        的 KeyError 风险死分支（两个分支都 return QUERY）；现已收敛到共享模块。
        """
        return _shared_conflict_guard(query, proposed, votes)

    def _engram_vote(self, features: np.ndarray, final_conf: float) -> Tuple[Optional[IntentType], float]:
        """读取 Engram 外部记忆，返回 (意向, 置信度) 或 (None, 0.0)。

        记忆检索失败/无记忆时返回 (None, 0.0)，调用方据此把记忆当作
        「无补充」处理，绝不影响主体投票结果。

        返回必须是可解包的二元组：调用处是
        ``eng_intent, eng_conf = self._engram_vote(...)``，直接 return None
        会在「无记忆」这条最常见路径上抛 TypeError。
        """
        if self.engram is None:
            return None, 0.0
        try:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(np.asarray(features, dtype=np.float32)).unsqueeze(0)
                cls_hidden = self.q_network.transformer(features_tensor)[:, 0]
                conf_tensor = torch.FloatTensor([[float(final_conf)]])
                enhanced = self.engram.read(
                    cls_hidden, features_tensor.squeeze(0).unsqueeze(0), conf_tensor
                )
                enhanced_q = self.q_network.advantage_stream(
                    self.q_network.value_stream(enhanced) + enhanced
                )
                enhanced_probs = F.softmax(enhanced_q, dim=-1).squeeze(0).numpy()
                eng_action = int(enhanced_probs.argmax())
                return self.action_map[eng_action], float(enhanced_probs[eng_action])
        except Exception:
            return None, 0.0

    def _safe_fallback(self, query: str) -> Optional[IntentType]:
        q = query.lower().strip()
        if not q:
            return IntentType.VAGUE

        # 问候语判定必须让位给明确的文件操作指令：模板里常把正文（如「你好」）
        # 拼进 query —— 「添加文件 notes.txt 内容为 你好」若先命中问候词会被
        # 误判成 vague，真正意图 add_file 被丢掉。
        op_kws = WRITE_KEYWORDS
        if any(w in q for w in ["你好", "嗨", "在吗", "你是谁", "hello", "hi", "hey",
                                  "谢谢", "好的", "嗯", "哈哈", "无聊", "讲个笑话",
                                  "天气", "你是AI", "随便聊", "你能做什么", "介绍自己"]) \
                and not any(w in q for w in op_kws):
            return IntentType.VAGUE
        if len(q) <= 2 and not any(c in q for c in "搜索查询删除添加修改"):
            return IntentType.VAGUE

        # 文件分析/扫描/报告检测（闸门词唯一定义在 intent_shared，
        # 与语义护栏、marl 的 rule_keywords 同口径）
        if any(kw in q for kw in ANALYSIS_GATE_KEYWORDS):
            has_forbidden = any(kw in q for kw in FORBIDDEN_MODIFY_KEYWORDS)

            if has_forbidden:
                return IntentType.FILE_ANALYSIS
            elif any(kw in q for kw in SCAN_KEYWORDS + ("scan",)):
                return IntentType.FILE_SCAN
            elif any(kw in q for kw in REPORT_KEYWORDS + ("review", "审查")):
                return IntentType.REPORT_GENERATE
            else:
                return IntentType.DEEP_SEARCH

        # 长指令检测
        if len(q) > 30:
            if any(kw in q for kw in DEEP_ANALYSIS_KEYWORDS):
                return IntentType.DEEP_SEARCH
            if any(kw in q for kw in SEARCH_KEYWORDS):
                return IntentType.QUERY

        if any(w in q for w in DEEP_ANALYSIS_KEYWORDS):
            return IntentType.DEEP_SEARCH
        if any(w in q for w in ADD_FILE_KEYWORDS):
            return IntentType.ADD_FILE
        if any(w in q for w in DELETE_KEYWORDS):
            return IntentType.DELETE_FILE
        if any(w in q for w in MODIFY_FILE_KEYWORDS):
            return IntentType.MODIFY_FILE
        if any(w in q for w in REFACTOR_KEYWORDS):
            return IntentType.REFACTOR
        if any(w in q for w in SEARCH_KEYWORDS):
            return IntentType.QUERY
        if any(w in q for w in FILENAME_QUERY_KEYWORDS):
            return IntentType.FILENAME_QUERY

        return IntentType.QUERY

    # ---------------------------------------------------------- 路由
    def route(self, intent: IntentType) -> RoutingPath:
        """意图 → 路由路径（唯一实现在 intent_shared.route_intent）。"""
        return _shared_route_intent(intent)

    def _detect_readonly_guard(self, query: str) -> bool:
        """检测用户是否明确要求只读操作（与 marl_agent 共用同一实现）。"""
        return detect_readonly_guard(query)

    # ---------------------------------------------------------- 奖励
    def compute_reward(self, outcome: str, state: AgentState = None) -> float:
        """按澄清深度折算后的奖励（state=None 时退回平表，供离线训练使用）。"""
        return round_reward(outcome, self.REWARD_MAP, state)

    # ---------------------------------------------------------- 学习
    def store_experience(self, state_vec, action: int, reward: float,
                         next_state_vec, done: bool):
        self.replay_buffer.push(state_vec, action, reward, next_state_vec, done)

    def learn(self, reward: float, state: AgentState, done: bool):
        state.add_reward(reward)
        if self.last_action is None or self.last_features is None:
            return
        query = state.dialog_history[-1].user_query if state.dialog_history else ""
        next_features = self._extract_features(query, state)
        self.store_experience(self.last_features, self.last_action, reward, next_features, done)
        self.reward_history.append(reward)
        self.train_step()

        # Engram 写入: 高置信度时积累经验
        if len(self.intent_history) > 0:
            last_action, last_conf = self.intent_history[-1]
            if last_conf > 0.7:
                with torch.no_grad():
                    feat_tensor = torch.FloatTensor(self.last_features).unsqueeze(0)
                    cls_hidden = self.q_network.transformer(feat_tensor)[:, 0]
                    conf_tensor = torch.FloatTensor([[last_conf]])
                    self.engram.write(cls_hidden, feat_tensor, conf_tensor)

    def _current_tau(self) -> float:
        """软更新系数调度：tau_init → tau_base 线性退火，之后固定 tau_base。"""
        if self.tau_anneal_steps <= 0:
            return self.tau_base
        frac = min(1.0, self.training_step / float(self.tau_anneal_steps))
        return self.tau_init + (self.tau_base - self.tau_init) * frac

    def _soft_update_target(self, tau: float = None) -> None:
        tau = self.tau if tau is None else tau
        with torch.no_grad():
            for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        if not batch:
            return

        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.FloatTensor([t[4] for t in batch])
        weights_t = torch.FloatTensor(weights)

        # --- DQN Loss (Double DQN) ---
        # 训练时同时获取 Q-values 和 NSP 预测
        current_q, nsp_pred = self.q_network.forward_with_nsp(states, actions)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (current_q.gather(1, actions.unsqueeze(1)).squeeze(1) - target_q).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        dqn_loss = (weights_t * F.smooth_l1_loss(
            current_q.gather(1, actions.unsqueeze(1)).squeeze(1),
            target_q, reduction='none'
        )).mean()

        # --- NSP Loss (Next State Prediction) ---
        # 目标: 从 next_state 编码得到实际的下一状态嵌入 (stop gradient)
        with torch.no_grad():
            target_next_tokens = self.q_network.transformer(next_states, task=1)
            target_next_state_embed = self.q_network.pool(target_next_tokens[:, 0])

        nsp_loss = self.q_network.compute_nsp_loss(nsp_pred, target_next_state_embed)

        # --- 分类损失 ---
        cls_loss = F.cross_entropy(
            self.classifier(states[:min(len(batch), self.batch_size)]),
            actions[:min(len(batch), self.batch_size)]
        )

        # --- 总损失 = DQN + λ * NSP + 0.3 * Classification ---
        total_loss = dqn_loss + self.nsp_lambda * nsp_loss + 0.3 * cls_loss

        # 爆炸/NaN 的 loss 直接跳过这一步，避免污染在线策略
        if not torch.isfinite(total_loss):
            self._soft_update_target(self._current_tau())
            return

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.classifier.parameters()), 1.0
        )
        self.optimizer.step()
        self.training_step += 1

        self._soft_update_target(self._current_tau())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ------------------------------------------------- 蒸馏训练
    def distill_from_llm(self, llm_labels: Dict[str, str], num_augment: int = 5):
        """从 LLM 标签蒸馏：真实句子 → LLM 标注 → 分类头有监督训练。"""
        if not llm_labels:
            return

        print(f"\n  [蒸馏] 从 {len(llm_labels)} 条 LLM 标签训练分类头...")
        intent_to_idx = self.reverse_map

        X, y = [], []
        for text, intent_str in llm_labels.items():
            try:
                label_idx = intent_to_idx[IntentType(intent_str)]
            except (ValueError, KeyError):
                continue

            stub = AgentState()
            stub.increment_round()
            vec = self._extract_features(text, stub)
            X.append(vec)
            y.append(label_idx)

            for _ in range(num_augment):
                aug = self._augment(text)
                vec_aug = self._extract_features(aug, stub)
                X.append(vec_aug)
                y.append(label_idx)

        if not X:
            return

        X_t = torch.FloatTensor(np.array(X))
        y_t = torch.LongTensor(y)

        self.classifier.train()
        optimizer = optim.Adam(self.classifier.parameters(), lr=1e-3)
        for epoch in range(30):
            logits = self.classifier(X_t)
            loss = F.cross_entropy(logits, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = (self.classifier(X_t).argmax(dim=1) == y_t).float().mean().item()
        print(f"  [蒸馏] 完成，训练集准确率 {correct:.1%}，样本数 {len(X)}")

    def _augment(self, text: str) -> str:
        """简单数据增强：随机删字、加前后缀。"""
        prefixes = ["请", "帮我", "能不能", "我想", "帮忙"]
        suffixes = ["一下", "吧", "好吗", "谢谢", ""]
        if random.random() < 0.5 and len(text) > 3:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1:]
        if random.random() < 0.5:
            return random.choice(prefixes) + text + random.choice(suffixes)
        return text

    # ------------------------------------------------- CSV 数据集训练
    def train_from_csv(self, csv_path: str = None, epochs: int = 10, augment_factor: int = 3):
        """从 CSV 文件加载数据训练分类头 + DQN。

        CSV 格式: text,label (label 为 add_file/delete_file/... 等)
        """
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "intent_corpus.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "datasets", "intent_corpus.csv")
        if not os.path.exists(csv_path):
            print(f"  [CSV训练] 数据集不存在: {csv_path}")
            return

        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                label = row.get("label", "").strip()
                if text and label:
                    samples.append((text, label))

        if not samples:
            print("  [CSV训练] 数据集为空")
            return

        print(f"\n  [CSV训练] 加载 {len(samples)} 条样本，训练 {epochs} 轮...")

        value_to_intent = {it.value: it for it in IntentType}
        intent_to_idx = self.reverse_map

        X, y = [], []
        for text, label_str in samples:
            intent = value_to_intent.get(label_str)
            if intent is None:
                continue
            label_idx = intent_to_idx[intent]
            stub = AgentState()
            stub.increment_round()
            vec = self._extract_features(text, stub)
            X.append(vec)
            y.append(label_idx)

            for _ in range(augment_factor):
                aug = self._augment(text)
                vec_aug = self._extract_features(aug, stub)
                X.append(vec_aug)
                y.append(label_idx)

        if not X:
            print("  [CSV训练] 无有效样本")
            return

        X_t = torch.FloatTensor(np.array(X))
        y_t = torch.LongTensor(y)
        n = len(X)

        self.classifier.train()
        cls_optimizer = optim.Adam(self.classifier.parameters(), lr=1e-3)

        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            correct = 0
            batch_size = 64
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                bx, by = X_t[idx], y_t[idx]
                logits = self.classifier(bx)
                loss = F.cross_entropy(logits, by)
                cls_optimizer.zero_grad()
                loss.backward()
                cls_optimizer.step()
                total_loss += loss.item() * len(idx)
                correct += (logits.argmax(dim=1) == by).sum().item()

        self.classifier.train()
        with torch.no_grad():
            logits_all = self.classifier(X_t)
            train_acc = (logits_all.argmax(dim=1) == y_t).float().mean().item()

        print(f"  [CSV训练] 完成，训练集准确率 {train_acc:.1%}，样本 {n}")

        self.trained = True
        self.save()

    # ------------------------------------------------- 快速训练（兼容旧接口）
    def train_episode(self, num_episodes: int = 10):
        """快速自博弈训练（兼容 main.py 启动调用）。"""
        print(f"\n  [RL Training] 快速训练 {num_episodes} 回合...")
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            intents = list(self.TASK_TEMPLATES.keys())
            random.shuffle(intents)
            for task in intents:
                # 训练只用训练池；评估用留出池（见 evaluate），避免训练集=测试集
                pool = _shared_train_pool(self.TASK_TEMPLATES[task])
                f = random.choice(self.SAMPLE_FILES)
                g = random.choice([x for x in self.SAMPLE_FILES if x != f])
                query = random.choice(pool).format(f=f, g=g, c=self.SAMPLE_CONTENT)
                stub = AgentState()
                stub.increment_round()
                vec = self._extract_features(query, stub)
                if random.random() < self.epsilon:
                    action = random.randint(0, self.num_actions - 1)
                else:
                    with torch.no_grad():
                        q = self.q_network(torch.FloatTensor(vec).unsqueeze(0))
                        action = q.argmax().item()
                correct = self.reverse_map[task]
                reward = _shared_train_reward(action == correct)
                # 单步自博弈：一步即一个 episode。旧实现 done=correct，
                # 答错时 next_state == state 却仍 bootstrap，TD 目标退化。
                done = True
                total_reward += reward
                self.store_experience(vec, action, reward, vec, done)
                self.train_step()
            self.reward_history.append(total_reward)
            print(f"    Episode {episode}/{num_episodes}: Reward={total_reward:.1f}, ε={self.epsilon:.4f}")
        self.trained = True

    # ------------------------------------------------- 评估
    def evaluate(self) -> Dict[str, float]:
        """独立测试集 + 模板测试集的每类准确率。"""
        self.q_network.eval()
        self.classifier.eval()
        results = {}

        value_to_intent = {it.value: it for it in IntentType}

        with torch.no_grad():
            for label_str, text in self.EVAL_REAL_SAMPLES.items():
                clean_label = label_str.rstrip("_0123456789")
                intent = value_to_intent.get(clean_label)
                if intent is None:
                    continue
                stub = AgentState()
                stub.increment_round()
                vec = self._extract_features(text, stub)
                cls_probs = self._classify_probs(vec)
                q_vals = self.q_network(torch.FloatTensor(vec).unsqueeze(0))
                q_probs = F.softmax(q_vals, dim=-1).squeeze(0).numpy()
                combined = 0.5 * cls_probs + 0.5 * q_probs
                pred = int(combined.argmax())
                correct = pred == self.reverse_map[intent]
                results[text[:20]] = {
                    "predicted": self.action_map[pred].value,
                    "expected": intent.value,
                    "correct": correct,
                }

        with torch.no_grad():
            for task, all_templates in self.TASK_TEMPLATES.items():
                # 评估只跑留出模板（训练阶段不可见），否则训练集=测试集
                templates = _shared_eval_pool(all_templates)
                ok = 0
                n_eval = min(40, len(templates) * 5)
                for _ in range(n_eval):
                    f = random.choice(self.SAMPLE_FILES)
                    g = random.choice([x for x in self.SAMPLE_FILES if x != f])
                    query = random.choice(templates).format(f=f, g=g, c=self.SAMPLE_CONTENT)
                    stub = AgentState()
                    stub.increment_round()
                    vec = self._extract_features(query, stub)
                    cls_probs = self._classify_probs(vec)
                    q_vals = self.q_network(torch.FloatTensor(vec).unsqueeze(0))
                    q_probs = F.softmax(q_vals, dim=-1).squeeze(0).numpy()
                    combined = 0.5 * cls_probs + 0.5 * q_probs
                    pred = int(combined.argmax())
                    if pred == self.reverse_map[task]:
                        ok += 1
                results[f"tpl_{task.value}"] = ok / n_eval

        self.q_network.train()
        self.classifier.train()
        return results

    # ------------------------------------------------- 完整管线
    def process(self, query: str, state: AgentState):
        rewritten, entities, relations = self.query_rewriter.rewrite(query)
        state.query_rewrite_history.append(rewritten)

        intent, confidence = self.classify_intent(query, state)
        routing_path = self.route(intent)

        # 只读安全守卫（共享实现，marl_agent.process 同样调用）
        pre_guard_intent = intent
        intent, readonly_downgraded = apply_readonly_guard(query, intent)
        if readonly_downgraded:
            print(f"  [安全守卫] 检测到只读请求，将 {pre_guard_intent.value} 降级为查询")
            routing_path = self.route(intent)
            confidence = max(confidence, 0.5)

        sub_intent = None
        if intent == IntentType.QUERY:
            if self.last_llm_sub_intent:
                try:
                    sub_intent = QuerySubIntent(self.last_llm_sub_intent)
                except ValueError:
                    sub_intent = self.query_rewriter.classify_query_subintent(query)
            else:
                sub_intent = self.query_rewriter.classify_query_subintent(query)
            state.current_sub_intent = sub_intent
            routing_path = self._get_query_routing_path(sub_intent)

        slots = self.query_rewriter.extract_slots(query)
        if self.last_llm_slots:
            slots.update(self.last_llm_slots)
        state.slots = slots

        clarifications = []
        if intent == IntentType.VAGUE:
            clarifications = self.query_rewriter.generate_clarification(query)
            # 置信度期望决策：低于 p* 才牺牲奖励继续询问，否则冒答错风险回答
            if not clarifications or not should_clarify(confidence, state):
                intent = IntentType.QUERY
                routing_path = self.route(intent)
                confidence = max(confidence, 0.4)
        elif intent in {IntentType.ADD_FILE, IntentType.DELETE_FILE, IntentType.MODIFY_FILE}:
            target = self.last_llm_target or self.query_rewriter.extract_target(query)
            if not target:
                clarifications = self.query_rewriter.generate_clarification(query)

        return intent, routing_path, confidence, clarifications

    def _get_query_routing_path(self, sub_intent: QuerySubIntent) -> RoutingPath:
        """根据子意图获取路由路径（唯一实现在 intent_shared.query_routing_path）。"""
        return _shared_query_routing_path(sub_intent)

    def get_intent_label(self, intent: IntentType) -> str:
        """意图 → 展示用短标签（唯一实现在 intent_shared.intent_label）。"""
        return intent_label(intent)
