"""Multi-Agent Reinforcement Learning (MARL) Framework for Intent Classification.

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        Meta-Agent (协调器)                          │
  │   - 收集各Agent的预测结果                                           │
  │   - 基于置信度加权仲裁                                              │
  │   - 处理冲突和边界情况                                              │
  └─────────┬───────────────┬───────────────┬───────────────────────────┘
            │               │               │
            ▼               ▼               ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ IntentAgent  │ │SubIntentAgent│ │  SlotAgent   │
  │ (主意图分类)  │ │ (子意图分类)  │ │ (槽位提取)   │
  │ 7类意图       │ │ 9类子意图     │ │ 多任务提取   │
  └──────────────┘ └──────────────┘ └──────────────┘

Design Principles:
  - Each agent is a specialist in its domain
  - Agents share semantic features but have separate Q-networks
  - MetaAgent learns to weight each agent's contribution
  - Joint training with shared reward signal
"""

import os
import random
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field

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

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MARL_MODEL_PATH = os.path.join(MODEL_DIR, "marl_agents.pth")


# ================================================================
#  Shared Feature Extractor (same as original)
# ================================================================
class SemanticFeatureExtractor:
    """Character n-gram hash + random projection -> 256-dim dense vector."""

    HASH_DIM = 2048
    PROJ_DIM = 256

    def __init__(self):
        import hashlib
        self._hashlib = hashlib
        rng = np.random.RandomState(42)
        self._proj = rng.randn(self.HASH_DIM, self.PROJ_DIM).astype(np.float32) * (1.0 / np.sqrt(self.HASH_DIM))

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
#  Specialized Agent Networks
# ================================================================
class IntentQNetwork(nn.Module):
    """Q-Network for primary intent classification (7 classes)."""

    def __init__(self, state_dim: int = 268, hidden_dim: int = 192, num_intents: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_intents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SubIntentQNetwork(nn.Module):
    """Q-Network for sub-intent classification (9 classes)."""

    def __init__(self, state_dim: int = 268, hidden_dim: int = 192, num_sub_intents: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_sub_intents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlotExtractionNetwork(nn.Module):
    """Multi-task network for slot extraction (file_path, image_path, language, etc.)."""

    def __init__(self, state_dim: int = 268, hidden_dim: int = 192, num_slots: int = 5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Multiple output heads for different slot types
        self.file_head = nn.Linear(hidden_dim // 2, 2)      # has_file / no_file
        self.image_head = nn.Linear(hidden_dim // 2, 2)     # has_image / no_image
        self.lang_head = nn.Linear(hidden_dim // 2, 7)      # 7 languages
        self.time_head = nn.Linear(hidden_dim // 2, 6)      # 6 time ranges
        self.format_head = nn.Linear(hidden_dim // 2, 5)    # 5 formats

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(x)
        # Split for different heads
        part1 = shared[:, :shared.shape[1]//2]
        part2 = shared[:, shared.shape[1]//2:]
        return {
            "file": F.softmax(self.file_head(part1), dim=-1),
            "image": F.softmax(self.image_head(part1), dim=-1),
            "language": F.softmax(self.lang_head(part2), dim=-1),
            "time": F.softmax(self.time_head(part2), dim=-1),
            "format": F.softmax(self.format_head(part2), dim=-1),
        }


# ================================================================
#  Individual Agents
# ================================================================
class IntentAgent:
    """Specialist agent for primary intent classification."""

    INTENT_LIST = [
        IntentType.ADD_FILE, IntentType.DELETE_FILE, IntentType.MODIFY_FILE,
        IntentType.REFACTOR, IntentType.QUERY, IntentType.DEEP_SEARCH, IntentType.VAGUE,
    ]

    def __init__(self, feature_dim: int = 268):
        self.feature_dim = feature_dim
        self.q_network = IntentQNetwork(feature_dim, 192, len(self.INTENT_LIST))
        self.target_network = IntentQNetwork(feature_dim, 192, len(self.INTENT_LIST))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)

        self.action_map = {i: intent for i, intent in enumerate(self.INTENT_LIST)}
        self.reverse_map = {intent: i for i, intent in enumerate(self.INTENT_LIST)}

        # Rule-based keywords for fallback
        self.rule_keywords = {
            IntentType.ADD_FILE: ["添加", "新增", "创建", "新建", "写入", "保存", "生成", "add", "create"],
            IntentType.DELETE_FILE: ["删除", "移除", "清理", "删掉", "remove", "delete"],
            IntentType.MODIFY_FILE: ["修改", "更新", "编辑", "改成", "modify", "edit", "update"],
            IntentType.REFACTOR: ["整理", "重构", "归类", "归档", "重命名", "refactor", "organize"],
            IntentType.QUERY: ["查", "找", "搜索", "查询", "检索", "怎么", "什么", "search", "find",
                                "识别", "截屏", "截图", "ocr", "文字", "输出", "提取", "读取", "查看"],
            IntentType.DEEP_SEARCH: ["对比", "比较", "分析", "差异", "综合", "总结", "analyze"],
        }

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Tuple[IntentType, float, np.ndarray]:
        """Predict intent with confidence and raw Q-values."""
        self.q_network.eval()
        x = torch.FloatTensor(features).unsqueeze(0)
        q_values = self.q_network(x).squeeze(0).numpy()
        probs = F.softmax(torch.FloatTensor(q_values), dim=-1).numpy()

        action_idx = int(probs.argmax())
        confidence = float(probs[action_idx])
        intent = self.action_map[action_idx]

        self.q_network.train()
        return intent, confidence, probs

    def rule_predict(self, query: str) -> Optional[IntentType]:
        """Rule-based prediction as backup."""
        q = query.lower()
        scores = {}
        
        for intent, keywords in self.rule_keywords.items():
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in q:
                    idx = q.find(kw_lower)
                    if idx == 0:
                        score += 2
                    else:
                        char_before = q[idx - 1]
                        if char_before in ' ,;:!?。，；：！？':
                            score += 2
                        else:
                            score += 1
            if score > 0:
                scores[intent] = score

        if not scores:
            return None
        return max(scores, key=scores.get)

    def compute_reward(self, predicted: IntentType, actual: IntentType, success: bool) -> float:
        """Compute reward for this agent's prediction."""
        if predicted == actual:
            return 5.0 if success else 2.0
        else:
            return -3.0


class SubIntentAgent:
    """Specialist agent for sub-intent classification (under QUERY)."""

    SUBINTENT_LIST = [
        QuerySubIntent.QUERY_METADATA, QuerySubIntent.QUERY_CONTENT,
        QuerySubIntent.QUERY_OCR, QuerySubIntent.QUERY_SUMMARY,
        QuerySubIntent.QUERY_TRANSLATE, QuerySubIntent.QUERY_EXTRACT,
        QuerySubIntent.QUERY_SEARCH, QuerySubIntent.QUERY_SYSTEM,
        QuerySubIntent.QUERY_OTHER,
    ]

    def __init__(self, feature_dim: int = 268):
        self.feature_dim = feature_dim
        self.q_network = SubIntentQNetwork(feature_dim, 192, len(self.SUBINTENT_LIST))
        self.target_network = SubIntentQNetwork(feature_dim, 192, len(self.SUBINTENT_LIST))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)

        self.action_map = {i: sub for i, sub in enumerate(self.SUBINTENT_LIST)}
        self.reverse_map = {sub: i for i, sub in enumerate(self.SUBINTENT_LIST)}

        # Priority keywords for rule-based backup
        self.priority_keywords = {
            QuerySubIntent.QUERY_OCR: ["识别", "ocr", "文字识别", "截屏", "截图"],
            QuerySubIntent.QUERY_TRANSLATE: ["翻译", "translate", "译成"],
            QuerySubIntent.QUERY_SUMMARY: ["总结", "摘要", "概括", "简述"],
            QuerySubIntent.QUERY_EXTRACT: ["提取", "抽出", "抽取"],
            QuerySubIntent.QUERY_METADATA: ["信息", "大小", "修改时间"],
            QuerySubIntent.QUERY_CONTENT: ["内容", "读取", "查看内容"],
            QuerySubIntent.QUERY_SEARCH: ["搜索", "查找", "检索"],
            QuerySubIntent.QUERY_SYSTEM: ["系统", "程序", "软件"],
        }

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Tuple[QuerySubIntent, float, np.ndarray]:
        """Predict sub-intent with confidence and raw Q-values."""
        self.q_network.eval()
        x = torch.FloatTensor(features).unsqueeze(0)
        q_values = self.q_network(x).squeeze(0).numpy()
        probs = F.softmax(torch.FloatTensor(q_values), dim=-1).numpy()

        action_idx = int(probs.argmax())
        confidence = float(probs[action_idx])
        sub_intent = self.action_map[action_idx]

        self.q_network.train()
        return sub_intent, confidence, probs

    def rule_predict(self, query: str) -> Optional[QuerySubIntent]:
        """Rule-based prediction with priority matching."""
        q = query.lower()
        # Priority match first
        for sub_intent, keywords in self.priority_keywords.items():
            for kw in keywords:
                if kw.lower() in q:
                    return sub_intent
        return None

    def compute_reward(self, predicted: QuerySubIntent, actual: QuerySubIntent) -> float:
        """Compute reward for this agent's prediction."""
        return 3.0 if predicted == actual else -2.0


class SlotAgent:
    """Specialist agent for slot extraction."""

    LANGUAGES = ["英文", "中文", "日文", "韩文", "法文", "德文", "其他"]
    TIME_RANGES = ["recent", "today", "yesterday", "this_week", "this_month", "other"]
    FORMATS = ["json", "csv", "markdown", "table", "other"]

    def __init__(self, feature_dim: int = 268):
        self.feature_dim = feature_dim
        self.network = SlotExtractionNetwork(feature_dim, 192, 5)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)

        # Regex patterns for extraction
        import re
        self.file_pattern = re.compile(r"[\w.\-\\/]+\.[A-Za-z0-9]{1,6}", re.IGNORECASE)
        self.image_pattern = re.compile(r"[\w.\-\\/]+\.(png|jpg|jpeg|bmp|tiff|gif)", re.IGNORECASE)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Dict[str, Tuple[str, float]]:
        """Predict all slots with confidence."""
        self.network.eval()
        x = torch.FloatTensor(features).unsqueeze(0)
        outputs = self.network(x)

        results = {}
        # File slot
        file_probs = outputs["file"][0].numpy()
        has_file = bool(file_probs.argmax())
        results["file_path"] = ("present" if has_file else "absent", float(file_probs.max()))

        # Image slot
        img_probs = outputs["image"][0].numpy()
        has_image = bool(img_probs.argmax())
        results["image_path"] = ("present" if has_image else "absent", float(img_probs.max()))

        # Language slot
        lang_probs = outputs["language"][0].numpy()
        lang_idx = int(lang_probs.argmax())
        results["language"] = (self.LANGUAGES[lang_idx], float(lang_probs[lang_idx]))

        # Time slot
        time_probs = outputs["time"][0].numpy()
        time_idx = int(time_probs.argmax())
        results["time_range"] = (self.TIME_RANGES[time_idx], float(time_probs[time_idx]))

        # Format slot
        fmt_probs = outputs["format"][0].numpy()
        fmt_idx = int(fmt_probs.argmax())
        results["format"] = (self.FORMATS[fmt_idx], float(fmt_probs[fmt_idx]))

        self.network.train()
        return results

    def rule_extract(self, query: str) -> Dict[str, Optional[str]]:
        """Rule-based slot extraction."""
        slots = {}

        # File path
        file_matches = self.file_pattern.findall(query)
        slots["file_path"] = file_matches[0] if file_matches else None

        # Image path
        img_matches = self.image_pattern.findall(query)
        slots["image_path"] = img_matches[0] if img_matches else None

        # Language
        q = query.lower()
        for lang in ["英文", "中文", "日文", "韩文", "法文", "德文"]:
            if lang in q:
                slots["language"] = lang
                break

        # Time range
        if any(kw in q for kw in ["最近", "刚才", "最新"]):
            slots["time_range"] = "recent"
        elif "今天" in q:
            slots["time_range"] = "today"
        elif "昨天" in q:
            slots["time_range"] = "yesterday"

        # Format
        for fmt in ["json", "csv", "markdown", "md格式", "表格"]:
            if fmt in q:
                slots["format"] = fmt.replace("md格式", "markdown").replace("表格", "table")
                break

        return slots


# ================================================================
#  Meta-Agent (Coordinator)
# ================================================================
class MetaAgent:
    """Meta-agent that coordinates all specialist agents and makes final decisions."""

    def __init__(self, feature_dim: int = 268):
        self.feature_dim = feature_dim

        # Initialize specialist agents
        self.intent_agent = IntentAgent(feature_dim)
        self.subintent_agent = SubIntentAgent(feature_dim)
        self.slot_agent = SlotAgent(feature_dim)

        # Initialize shared components
        self.feature_extractor = SemanticFeatureExtractor()
        self.query_rewriter = QueryRewriter()
        self.llm_classifier = LLMIntentClassifier()

        # LLM results storage
        self.last_llm_target = None
        self.last_llm_sub_intent = None
        self.last_llm_slots = {}

        # Learning parameters
        self.epsilon = 0.3
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 32

        # Experience buffer (shared across agents)
        self.replay_buffer = []
        self.max_buffer_size = 10000
        self.training_step = 0

        # Load model if exists
        self.model_path = DEFAULT_MARL_MODEL_PATH
        self.loaded_from_disk = False
        self._auto_load()

    def _auto_load(self):
        """Load saved model if exists."""
        if not os.path.exists(self.model_path):
            return
        try:
            self.load(self.model_path)
            self.loaded_from_disk = True
            print(f"\n  [MARL] 已加载模型 {self.model_path} (ε={self.epsilon:.4f})")
        except Exception as e:
            self.loaded_from_disk = False
            print(f"  [MARL] 加载失败({e})，重新初始化。")

    def save(self, path: str = None):
        """Save all agent models."""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "intent_q": self.intent_agent.q_network.state_dict(),
            "intent_target": self.intent_agent.target_network.state_dict(),
            "intent_optimizer": self.intent_agent.optimizer.state_dict(),
            "subintent_q": self.subintent_agent.q_network.state_dict(),
            "subintent_target": self.subintent_agent.target_network.state_dict(),
            "subintent_optimizer": self.subintent_agent.optimizer.state_dict(),
            "slot_net": self.slot_agent.network.state_dict(),
            "slot_optimizer": self.slot_agent.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
        }, path)
        print(f"  [MARL] 已保存 → {path}")

    def load(self, path: str):
        """Load all agent models."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "intent_q" in ckpt:
            self.intent_agent.q_network.load_state_dict(ckpt["intent_q"])
        if "intent_target" in ckpt:
            self.intent_agent.target_network.load_state_dict(ckpt["intent_target"])
        if "intent_optimizer" in ckpt:
            self.intent_agent.optimizer.load_state_dict(ckpt["intent_optimizer"])
        if "subintent_q" in ckpt:
            self.subintent_agent.q_network.load_state_dict(ckpt["subintent_q"])
        if "subintent_target" in ckpt:
            self.subintent_agent.target_network.load_state_dict(ckpt["subintent_target"])
        if "subintent_optimizer" in ckpt:
            self.subintent_agent.optimizer.load_state_dict(ckpt["subintent_optimizer"])
        if "slot_net" in ckpt:
            self.slot_agent.network.load_state_dict(ckpt["slot_net"])
        if "slot_optimizer" in ckpt:
            self.slot_agent.optimizer.load_state_dict(ckpt["slot_optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.training_step = ckpt.get("training_step", self.training_step)

    def _extract_features(self, query: str, state: AgentState) -> np.ndarray:
        """Extract 268-dim features (256 semantic + 12 state)."""
        sem = self.feature_extractor.extract(query)
        state_vec = state.get_state_vector()
        return np.concatenate([sem, state_vec]).astype(np.float32)

    def classify_intent(self, query: str, state: AgentState) -> Tuple[IntentType, float]:
        """Main entry: Multi-agent classification with meta-arbitration.

        Returns:
            (intent, confidence) - final classification result
        """
        self.last_llm_target = None
        self.last_llm_sub_intent = None
        self.last_llm_slots = {}

        # Step 1: Get LLM result (if available)
        llm_intent = None
        llm_confidence = 0.0
        llm_result = self.llm_classifier.classify(query)
        if llm_result:
            intent_value, target, confidence, sub_intent, slots = llm_result
            try:
                llm_intent = IntentType(intent_value)
                llm_confidence = confidence
                self.last_llm_target = target
                self.last_llm_sub_intent = sub_intent
                self.last_llm_slots = slots
            except ValueError:
                pass

        # Step 2: Extract features
        features = self._extract_features(query, state)

        # Step 3: IntentAgent prediction (with epsilon-greedy)
        if random.random() < self.epsilon:
            # Random exploration
            intent_action = random.randint(0, len(self.intent_agent.INTENT_LIST) - 1)
            intent_from_qnet = self.intent_agent.action_map[intent_action]
            intent_conf_qnet = 1.0 / len(self.intent_agent.INTENT_LIST)
        else:
            intent_from_qnet, intent_conf_qnet, _ = self.intent_agent.predict(features)

        # Step 4: Rule-based prediction
        intent_from_rule = self.intent_agent.rule_predict(query)

        # Step 5: Meta-arbitration
        candidates = []

        # Add LLM prediction (highest weight)
        if llm_intent is not None:
            candidates.append((llm_intent, min(1.0, llm_confidence * 1.2), "llm"))

        # Add Q-Network prediction
        candidates.append((intent_from_qnet, intent_conf_qnet, "qnet"))

        # Add Rule prediction (higher weight when LLM unavailable)
        if intent_from_rule is not None:
            rule_weight = 0.55 if llm_intent is not None else 0.75
            candidates.append((intent_from_rule, rule_weight, "rule"))

        # Aggregate votes
        votes: Dict[IntentType, Dict] = {}
        for intent, score, source in candidates:
            if intent not in votes:
                votes[intent] = {"score": 0.0, "sources": []}
            votes[intent]["score"] += score
            votes[intent]["sources"].append(source)

        # Average scores per intent
        for intent, v in votes.items():
            v["score"] /= len(v["sources"])

        # Select winner
        final_intent = max(votes, key=lambda k: votes[k]["score"])
        final_conf = votes[final_intent]["score"]

        # Semantic conflict guard
        final_intent = self._semantic_conflict_guard(query, final_intent, votes)

        # Store for learning
        self._last_features = features
        self._last_intent = final_intent
        self._last_confidence = final_conf

        return final_intent, final_conf

    def _semantic_conflict_guard(self, query: str, proposed: IntentType,
                                  votes: Dict[IntentType, Dict]) -> IntentType:
        """Guard against semantic conflicts (e.g., search keywords but predicted DELETE)."""
        q = query.lower().strip()

        search_kws = ["找", "查", "搜索", "查询", "检索", "怎么", "什么", "哪里",
                       "search", "find", "查看", "看看", "在哪", "定位",
                       "识别", "截屏", "截图", "ocr", "文字", "输出", "提取", "读取"]
        has_search = any(kw in q for kw in search_kws)

        delete_kws = ["删除", "移除", "清理", "删掉", "干掉", "remove", "delete"]
        has_delete_kw = any(kw in q for kw in delete_kws)

        if has_search and not has_delete_kw:
            if proposed == IntentType.DELETE_FILE:
                return IntentType.QUERY
            if proposed == IntentType.REFACTOR:
                return IntentType.QUERY

        return proposed

    def get_sub_intent(self, query: str, features: np.ndarray) -> QuerySubIntent:
        """Get sub-intent for QUERY type."""
        # Rule-based first (high precision for specific patterns)
        rule_result = self.subintent_agent.rule_predict(query)
        if rule_result is not None:
            return rule_result

        # Q-Network prediction
        if random.random() < self.epsilon:
            sub_action = random.randint(0, len(self.subintent_agent.SUBINTENT_LIST) - 1)
            return self.subintent_agent.action_map[sub_action]
        else:
            sub_intent, _, _ = self.subintent_agent.predict(features)
            return sub_intent

    def get_slots(self, query: str, features: np.ndarray) -> Dict[str, any]:
        """Get extracted slots."""
        # Neural prediction
        neural_slots = self.slot_agent.predict(features)

        # Rule-based extraction
        rule_slots = self.slot_agent.rule_extract(query)

        # Merge: prefer rule results when confident, otherwise use neural
        final_slots = {}
        for key in ["file_path", "image_path", "language", "time_range", "format"]:
            rule_val = rule_slots.get(key)
            neural_val, neural_conf = neural_slots.get(key, (None, 0.0))

            if rule_val is not None:
                final_slots[key] = rule_val
            elif neural_conf > 0.6 and neural_val != "absent":
                final_slots[key] = neural_val

        return final_slots

    def process(self, query: str, state: AgentState) -> Tuple[IntentType, any, float, List[str]]:
        """Full MARL processing pipeline.

        Returns:
            (intent, routing_path, confidence, clarifications)
        """
        # Query rewriting
        rewritten, entities, relations = self.query_rewriter.rewrite(query)
        state.query_rewrite_history.append(rewritten)

        # Intent classification
        intent, confidence = self.classify_intent(query, state)

        # Get routing path based on intent
        routing_path = self._get_routing_path(intent)

        # Sub-intent for QUERY
        sub_intent = None
        if intent == IntentType.QUERY:
            features = self._extract_features(query, state)

            if self.last_llm_sub_intent:
                try:
                    sub_intent = QuerySubIntent(self.last_llm_sub_intent)
                except ValueError:
                    sub_intent = self.get_sub_intent(query, features)
            else:
                sub_intent = self.get_sub_intent(query, features)

            state.current_sub_intent = sub_intent
            routing_path = self._get_query_routing_path(sub_intent)

        # Slot extraction
        features = self._extract_features(query, state)
        slots = self.get_slots(query, features)
        if self.last_llm_slots:
            slots.update(self.last_llm_slots)
        state.slots = slots

        # Clarification handling
        clarifications = []
        if intent == IntentType.VAGUE:
            clarifications = self.query_rewriter.generate_clarification(query)
            if not clarifications or len(query.strip()) > 3:
                intent = IntentType.QUERY
                routing_path = self._get_routing_path(intent)
                confidence = max(confidence, 0.4)
        elif intent in {IntentType.ADD_FILE, IntentType.DELETE_FILE, IntentType.MODIFY_FILE}:
            target = self.last_llm_target or self.query_rewriter.extract_target(query)
            if not target:
                clarifications = self.query_rewriter.generate_clarification(query)

        return intent, routing_path, confidence, clarifications

    def _get_routing_path(self, intent: IntentType) -> any:
        """Get routing path for primary intent."""
        from agent_state import RoutingPath
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

    def _get_query_routing_path(self, sub_intent: QuerySubIntent) -> any:
        """Get routing path for query sub-intent."""
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
        }
        return routing_map.get(sub_intent, RoutingPath.SINGLE_VECTOR)

    def learn(self, reward: float, state: AgentState, done: bool):
        """Learn from experience - update all agents."""
        if not hasattr(self, '_last_features') or self._last_features is None:
            return

        query = state.dialog_history[-1].user_query if state.dialog_history else ""
        next_features = self._extract_features(query, state)

        # Store experience
        experience = {
            "state": self._last_features,
            "intent": self._last_intent,
            "reward": reward,
            "next_state": next_features,
            "done": done,
        }
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

        # Train if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _train_step(self):
        """Training step for all agents."""
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Prepare tensors
        states = torch.FloatTensor(np.array([e["state"] for e in batch]))
        next_states = torch.FloatTensor(np.array([e["next_state"] for e in batch]))
        rewards = torch.FloatTensor([e["reward"] for e in batch])
        dones = torch.FloatTensor([e["done"] for e in batch])

        # Train Intent Agent
        intent_actions = torch.LongTensor([
            self.intent_agent.reverse_map.get(e["intent"], 0) for e in batch
        ])
        self._train_q_network(
            self.intent_agent.q_network,
            self.intent_agent.target_network,
            self.intent_agent.optimizer,
            states, intent_actions, rewards, next_states, dones
        )

        # Train SubIntent Agent (only for QUERY intents)
        query_mask = torch.BoolTensor([e["intent"] == IntentType.QUERY for e in batch])
        if query_mask.any():
            query_states = states[query_mask]
            query_next = next_states[query_mask]
            query_rewards = rewards[query_mask]
            query_dones = dones[query_mask]

            # Use rule-based labels for sub-intent training
            subintent_actions = torch.LongTensor([
                self.subintent_agent.reverse_map.get(
                    self.subintent_agent.rule_predict(
                        state.dialog_history[-1].user_query if state.dialog_history else ""
                    ) or QuerySubIntent.QUERY_OTHER,
                    8
                )
                for state in [AgentState() for _ in range(query_mask.sum())]
            ])
            self._train_q_network(
                self.subintent_agent.q_network,
                self.subintent_agent.target_network,
                self.subintent_agent.optimizer,
                query_states, subintent_actions, query_rewards, query_next, query_dones
            )

        self.training_step += 1

    def _train_q_network(self, q_net, target_net, optimizer,
                          states, actions, rewards, next_states, dones):
        """Train a Q-network with DQN loss."""
        current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
            next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        optimizer.step()

        # Soft update target network
        for tp, p in zip(target_net.parameters(), q_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train_episode(self, num_episodes: int = 10):
        """Quick training with template data (for compatibility with original AgentRL).

        This method provides basic warm-up training using template queries.
        """
        print(f"\n  [MARL Training] 快速训练 {num_episodes} 回合...")

        # Template queries for each intent
        templates = {
            IntentType.ADD_FILE: [
                "添加文件 {f} 内容为 这是测试内容",
                "创建新文件 {f}",
                "帮我新建一个 {f}",
            ],
            IntentType.DELETE_FILE: [
                "删除文件 {f}",
                "移除 {f}",
                "把 {f} 删掉",
            ],
            IntentType.MODIFY_FILE: [
                "修改文件 {f} 内容为 新内容",
                "更新 {f}",
                "编辑 {f}",
            ],
            IntentType.REFACTOR: [
                "整理文件夹",
                "重构项目结构",
                "把文件按类型分类",
            ],
            IntentType.QUERY: [
                "搜索 强化学习",
                "查找 {f}",
                "查看这个文件",
            ],
            IntentType.DEEP_SEARCH: [
                "对比分析 {f} 和 {g}",
                "深度分析这个项目",
                "综合比较两者差异",
            ],
            IntentType.VAGUE: [
                "你好",
                "你是谁",
                "在吗",
            ],
        }

        sample_files = ["notes.txt", "config.yaml", "main.py", "readme.md"]

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0

            for intent, template_list in templates.items():
                for template in template_list:
                    f = random.choice(sample_files)
                    g = random.choice([x for x in sample_files if x != f])
                    query = template.format(f=f, g=g)

                    # Process query
                    state = AgentState()
                    state.increment_round()
                    predicted_intent, path, confidence, _ = self.process(query, state)

                    # Compute reward
                    reward = 3.0 if predicted_intent == intent else -1.0
                    total_reward += reward

                    # Learn
                    self.learn(reward, state, done=True)

            print(f"    Episode {episode}/{num_episodes}: Reward={total_reward:.1f}, ε={self.epsilon:.4f}")

        self.save()
        print("  [MARL Training] 完成并保存模型")

    def compute_reward(self, outcome: str) -> float:
        """Compute reward based on outcome key.

        Compatible with original AgentRL interface.
        """
        reward_map = {
            "add_success": 5.0, "delete_success": 5.0, "modify_success": 5.0,
            "refactor_success": 7.0, "query_success": 5.0, "deep_search_success": 8.0,
            "add_failure": -4.0, "delete_failure": -4.0, "modify_failure": -4.0,
            "refactor_failure": -4.0, "insufficient_evidence": -2.0,
            "correct_routing": 10.0, "wrong_routing": -5.0,
            "clarification_resolved": 7.0, "unnecessary_computation": -3.0,
            "timeout": -3.0, "no_evidence": -4.0, "evidence_sufficient": 8.0,
        }
        return reward_map.get(outcome, 0.0)

    def get_intent_label(self, intent: IntentType) -> str:
        """Get human-readable label for intent."""
        return SHORT_LABELS.get(intent, intent.value)


# ================================================================
#  Factory function
# ================================================================
def create_marl_agent() -> MetaAgent:
    """Create and return a MARL agent instance."""
    return MetaAgent()
