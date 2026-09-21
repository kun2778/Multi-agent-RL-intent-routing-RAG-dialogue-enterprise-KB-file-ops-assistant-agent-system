"""Intent Classification Agent v2 — 语义特征 + 分类蒸馏 + Double DQN。

架构:
  1. SemanticFeatureExtractor — 中文字符 n-gram 哈希 → 256 维稠密向量
  2. DuelingQNetwork — Dueling + Double DQN 结构
  3. PrioritizedReplayBuffer — 按 TD-error 比例采样
  4. ClassificationHead — 交叉熵分类 + LLM 知识蒸馏
  5. ConfidenceArbiter — LLM / 规则 / Q-网络 置信度仲裁
"""

import csv
import hashlib
import math
import os
import random
from typing import Dict, List, Tuple, Optional, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent_state import IntentType, QuerySubIntent, RoutingPath, AgentState, SHORT_LABELS, QUERY_SUBINTENT_LABELS
from query_rewriter import QueryRewriter
from llm_intent import LLMIntentClassifier

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "agent_rl.pth")


# ================================================================
#  1. 语义特征提取器 — 字符 n-gram 哈希 + 随机投影
# ================================================================
class SemanticFeatureExtractor:
    """中文/英文字符 n-gram 哈希 + 随机投影 → 256 维稠密向量。

    原理:
      - 提取 unigram + bigram + trigram（字符级）
      - 每个 n-gram 哈希到 HASH_DIM 维稀疏桶
      - 乘以固定的随机投影矩阵 → 256 维稠密向量
      - 无需训练，无需外部依赖，启动即用
    """

    HASH_DIM = 2048
    PROJ_DIM = 256

    def __init__(self):
        rng = np.random.RandomState(42)
        self._proj = rng.randn(self.HASH_DIM, self.PROJ_DIM).astype(np.float32) * (1.0 / math.sqrt(self.HASH_DIM))

    def _char_ngrams(self, text: str) -> List[str]:
        text = text.lower().strip()
        ngrams = []
        for n in (1, 2, 3):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i + n])
        return ngrams

    def _hash_to_bucket(self, s: str) -> int:
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
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
    """按 TD-error 比例采样的经验回放。"""

    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = 0.0001
        self.max_priority = 1.0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, state, action, reward, next_state, done):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))
        self._size = min(self._size + 1, self.tree.capacity)

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
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
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority ** (1.0 / self.alpha))
            self.tree.update(int(idx), priority)


# ================================================================
#  5. 主 Agent
# ================================================================
class AgentRL:
    """意图分类 Agent：分类头 + Dueling Double DQN + 蒸馏。

    特征:
      - 256 维语义特征（字符 n-gram 哈希） + 12 维状态向量 = 268 维输入
      - 分类头：交叉熵有监督训练（模板 + 蒸馏标签）
      - DQN：Double DQN + Dueling + Prioritized Replay
      - 置信度仲裁：LLM / 规则 / 网络 三方投票
    """

    INTENT_ORDER = [
        IntentType.ADD_FILE, IntentType.DELETE_FILE, IntentType.MODIFY_FILE,
        IntentType.REFACTOR, IntentType.QUERY, IntentType.DEEP_SEARCH, IntentType.VAGUE,
    ]
    FEATURE_DIM = 268

    REWARD_MAP: Dict[str, float] = {
        "add_success": 5.0, "delete_success": 5.0, "modify_success": 5.0,
        "refactor_success": 7.0, "query_success": 5.0, "deep_search_success": 8.0,
        "add_failure": -4.0, "delete_failure": -4.0, "modify_failure": -4.0,
        "refactor_failure": -4.0, "insufficient_evidence": -2.0,
        "correct_routing": 10.0, "wrong_routing": -5.0,
        "clarification_resolved": 7.0, "unnecessary_computation": -3.0,
        "timeout": -3.0, "no_evidence": -4.0, "evidence_sufficient": 8.0,
    }

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
    }
    SAMPLE_FILES = ["notes.txt", "config.yaml", "main.py", "app.json", "report.md", "readme.md", "data.csv"]
    SAMPLE_CONTENT = "这是一段用于演示的文档内容。"

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
    }

    def __init__(self, model_path: str = None):
        self.query_rewriter = QueryRewriter()
        self.state_dim = self.FEATURE_DIM
        self.num_actions = len(self.INTENT_ORDER)
        self.action_map: Dict[int, IntentType] = {i: it for i, it in enumerate(self.INTENT_ORDER)}
        self.reverse_map: Dict[IntentType, int] = {it: i for i, it in self.action_map.items()}

        self.feature_extractor = SemanticFeatureExtractor()
        self.classifier = ClassificationHead(self.state_dim, 128, self.num_actions)
        self.q_network = DuelingQNetwork(self.state_dim, 192, self.num_actions)
        self.target_network = DuelingQNetwork(self.state_dim, 192, self.num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())

        all_params = list(self.q_network.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.Adam(all_params, lr=5e-4)
        self.replay_buffer = PrioritizedReplayBuffer(50000)

        self.epsilon = 0.3
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.training_step = 0

        self.intent_history: List[Tuple[int, float]] = []
        self.reward_history: List[float] = []
        self.trained = False

        self.last_features: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

        self.llm_classifier = LLMIntentClassifier()
        self.last_llm_target: Optional[str] = None
        self.last_llm_sub_intent: Optional[str] = None
        self.last_llm_slots: Dict[str, str] = {}

        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.loaded_from_disk = False
        self._auto_load()

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

    def save(self, path: str = None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
            "trained": self.trained,
        }, path)
        print(f"  [持久化] 已保存 → {path} (步数 {self.training_step})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "q_network" in ckpt:
            self.q_network.load_state_dict(ckpt["q_network"])
        if "target_network" in ckpt:
            self.target_network.load_state_dict(ckpt["target_network"])
        if "classifier" in ckpt:
            self.classifier.load_state_dict(ckpt["classifier"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                pass
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.training_step = ckpt.get("training_step", self.training_step)
        self.trained = ckpt.get("trained", self.trained)

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

    # --------------------------------------------------- 置信度仲裁
    def classify_intent(self, query: str, state: AgentState) -> Tuple[IntentType, float]:
        """多方仲裁：LLM → 规则+Q网络+分类头 → 保底。

        冲突时用置信度决定最终意图。
        规则在无 LLM 时权重提升，且对语义冲突有纠错能力。
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
                rule_weight = 0.55 if llm_available else 0.75
                candidates.append((rule_intent, rule_weight, "rule"))
            except ValueError:
                pass

        votes: Dict[IntentType, Dict] = {}
        for intent, score, source in candidates:
            if intent not in votes:
                votes[intent] = {"score": 0.0, "sources": []}
            votes[intent]["score"] += score
            votes[intent]["sources"].append(source)

        for intent, v in votes.items():
            v["score"] /= len(v["sources"])

        final_intent = max(votes, key=lambda k: votes[k]["score"])
        final_conf = votes[final_intent]["score"]

        if final_conf < 0.25:
            safe = self._safe_fallback(query)
            if safe is not None:
                final_intent = safe
                final_conf = 0.4

        final_intent = self._semantic_conflict_guard(query, final_intent, votes)

        return final_intent, final_conf

    def _semantic_conflict_guard(self, query: str, proposed: IntentType,
                                  votes: Dict[IntentType, Dict]) -> IntentType:
        """语义冲突护栏：当关键词明确指向查询但模型预测为删除时，强制纠正。"""
        q = query.lower().strip()

        search_kws = ["找", "查", "搜索", "查询", "检索", "怎么", "什么", "哪里",
                       "search", "find", "查看", "看看", "在哪", "定位"]
        has_search = any(kw in q for kw in search_kws)

        delete_kws = ["删除", "移除", "清理", "删掉", "干掉", "remove", "delete"]
        has_delete_kw = any(kw in q for kw in delete_kws)

        if has_search and not has_delete_kw:
            if proposed == IntentType.DELETE_FILE:
                if "rule" in votes and votes[IntentType.QUERY]["score"] > 0:
                    return IntentType.QUERY
                return IntentType.QUERY

        return proposed

    def _safe_fallback(self, query: str) -> Optional[IntentType]:
        q = query.lower().strip()
        if not q:
            return IntentType.VAGUE

        if any(w in q for w in ["你好", "嗨", "在吗", "你是谁", "hello", "hi", "hey",
                                  "谢谢", "好的", "嗯", "哈哈", "无聊", "讲个笑话",
                                  "天气", "你是AI", "随便聊", "你能做什么", "介绍自己"]):
            return IntentType.VAGUE
        if len(q) <= 2 and not any(c in q for c in "搜索查询删除添加修改"):
            return IntentType.VAGUE

        if any(w in q for w in ["对比", "比较", "分析", "差异", "区别", "vs", "综合", "总结", "趋势", "影响", "评估"]):
            return IntentType.DEEP_SEARCH
        if any(w in q for w in ["添加", "新增", "创建", "新建", "写入", "保存", "生成", "add", "create", "new"]):
            return IntentType.ADD_FILE
        if any(w in q for w in ["删除", "移除", "清理", "删掉", "remove", "delete"]):
            return IntentType.DELETE_FILE
        if any(w in q for w in ["修改", "更新", "编辑", "改成", "modify", "edit", "update"]):
            return IntentType.MODIFY_FILE
        if any(w in q for w in ["整理", "重构", "归类", "归档", "重命名", "移动",
                                  "分一下类", "分类", "分类归类", "refactor", "organize"]):
            return IntentType.REFACTOR
        if any(w in q for w in ["查", "找", "搜索", "查询", "检索", "怎么", "什么", "哪里", "search", "find", "查看",
                                  "识别", "截屏", "截图", "ocr", "文字", "输出", "提取", "读取"]):
            return IntentType.QUERY

        return IntentType.QUERY

    # ---------------------------------------------------------- 路由
    def route(self, intent: IntentType) -> RoutingPath:
        routing_map = {
            IntentType.ADD_FILE: RoutingPath.FILE_ADD,
            IntentType.DELETE_FILE: RoutingPath.FILE_DELETE,
            IntentType.MODIFY_FILE: RoutingPath.FILE_MODIFY,
            IntentType.REFACTOR: RoutingPath.FILE_REFACTOR,
            IntentType.QUERY: RoutingPath.SINGLE_VECTOR,
            IntentType.DEEP_SEARCH: RoutingPath.DEEP_SEARCH,
            IntentType.VAGUE: RoutingPath.CLARIFICATION,
        }
        return routing_map[intent]

    # ---------------------------------------------------------- 奖励
    def compute_reward(self, outcome: str) -> float:
        return self.REWARD_MAP.get(outcome, 0.0)

    # ---------------------------------------------------------- 学习
    def store_experience(self, state_vec, action: int, reward: float,
                         next_state_vec, done: bool):
        self.replay_buffer.push(state_vec, action, reward, next_state_vec, done)

    def learn(self, reward: float, state: AgentState, done: bool):
        if self.last_action is None or self.last_features is None:
            return
        query = state.dialog_history[-1].user_query if state.dialog_history else ""
        next_features = self._extract_features(query, state)
        self.store_experience(self.last_features, self.last_action, reward, next_features, done)
        self.reward_history.append(reward)
        state.add_reward(reward)
        self.train_step()

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

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (current_q - target_q).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        cls_loss = F.cross_entropy(
            self.classifier(states[:min(len(batch), self.batch_size)]),
            actions[:min(len(batch), self.batch_size)]
        )
        total_loss = loss + 0.3 * cls_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.classifier.parameters()), 1.0
        )
        self.optimizer.step()
        self.training_step += 1

        for tp, p in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

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
                f = random.choice(self.SAMPLE_FILES)
                g = random.choice([x for x in self.SAMPLE_FILES if x != f])
                query = random.choice(self.TASK_TEMPLATES[task]).format(f=f, g=g, c=self.SAMPLE_CONTENT)
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
                reward = 5.0 if action == correct else -2.0
                done = action == correct
                if done:
                    reward += 10.0
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
            for task, templates in self.TASK_TEMPLATES.items():
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
            if not clarifications or len(query.strip()) > 3:
                intent = IntentType.QUERY
                routing_path = self.route(intent)
                confidence = max(confidence, 0.4)
        elif intent in {IntentType.ADD_FILE, IntentType.DELETE_FILE, IntentType.MODIFY_FILE}:
            target = self.last_llm_target or self.query_rewriter.extract_target(query)
            if not target:
                clarifications = self.query_rewriter.generate_clarification(query)

        return intent, routing_path, confidence, clarifications

    def _get_query_routing_path(self, sub_intent: QuerySubIntent) -> RoutingPath:
        """根据子意图获取路由路径。"""
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
        }
        return routing_map.get(sub_intent, RoutingPath.SINGLE_VECTOR)

    def get_intent_label(self, intent: IntentType) -> str:
        return SHORT_LABELS.get(intent, intent.value)
