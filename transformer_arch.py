"""Transformer-based Architecture with Looped Transformer, MoE, and Engram Memory.

替代原 MLP Q-network 的完整 Transformer 架构:
  1. InputTokenizer        — 268维向量 → 36个token
  2. MoELayer               — 8专家 Expert-Choice top-2, 无辅助损失负载均衡
  3. TaskAwareMoELayer      — MoE + 任务感知路由 (Q专家 vs NSP专家)
  4. LoopedTransformerBlock — Self-Attn + MoE-FFN (单个块)
  5. TaskAwareLoopedTransformerBlock — Self-Attn + TaskAwareMoE-FFN
  6. LoopedTransformer      — 多次循环复用同一块 (T=6)
  7. TaskAwareLoopedTransformer — 任务感知版
  8. EngramConditionalMemory — 8-slot条件记忆, 解决幻觉 (Agent层面使用)
  9. NSPHead                — 旧版 NSP 辅助头 (已弃用: 训练侧改用 nsp_latent.py,
                              本类仅为旧 checkpoint 的 nsp_head.* key 兼容保留, 不可删除)
  10. TransformerDuelingQNetwork        — 单agent Dueling Q-network
  11. TransformerDuelingQNetworkWithNSP — 单agent Q-network + NSP (nsp_head 已不参与训练)
  12. TransformerClassificationHead     — 单agent 分类头
  13. TransformerIntentQNetwork         — 多agent 主意图 Q-network
  14. TransformerSubIntentQNetwork      — 多agent 子意图 Q-network
  15. TransformerSlotExtractionNetwork  — 多agent 槽位提取
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
#  1. InputTokenizer — 268维 → token序列
# ================================================================
class InputTokenizer(nn.Module):
    """268/272维向量 → token序列 (35/36个特征token + 1个CLS)。

    分组策略:
      - 256维语义特征 → 32组, 每组8维
      - 12维状态特征 → 3组, 每组4维
      - 附加通道 (state_dim - 268, 须为4的倍数) → 每4维1组（如 MARL 的角色辅助4维）
      - 1个可学习CLS token (全局表征)

    state_dim=268 时参数与旧版完全一致（不新增参数），旧检查点可直接加载。
    """

    SEMANTIC_GROUPS = 32
    STATE_GROUPS = 3
    TOTAL_GROUPS = 35
    SEM_DIM = 256
    STATE_DIM = 12

    def __init__(self, d_model: int = 128, state_dim: int = 268):
        super().__init__()
        self.d_model = d_model
        self.state_dim = int(state_dim)
        extra = self.state_dim - self.SEM_DIM - self.STATE_DIM
        if extra < 0 or extra % 4 != 0:
            raise ValueError(
                f"state_dim={state_dim} 不支持：需 >= 268 且超出 268 的部分为 4 的倍数"
            )
        self.extra_groups = extra // 4
        self.semantic_proj = nn.Linear(8, d_model)
        self.state_proj = nn.Linear(4, d_model)
        self.aux_proj = nn.Linear(4, d_model) if self.extra_groups else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        semantic = x[:, :self.SEM_DIM].reshape(B, self.SEMANTIC_GROUPS, 8)
        rest = x[:, self.SEM_DIM:]
        state = rest[:, :self.STATE_DIM].reshape(B, self.STATE_GROUPS, 4)

        sem_tokens = self.semantic_proj(semantic)
        state_tokens = self.state_proj(state)

        cls = self.cls_token.expand(B, -1, -1)
        parts = [cls, sem_tokens, state_tokens]
        if self.extra_groups:
            aux = rest[:, self.STATE_DIM:self.STATE_DIM + 4 * self.extra_groups]
            parts.append(self.aux_proj(aux.reshape(B, self.extra_groups, 4)))
        tokens = torch.cat(parts, dim=1)
        return self.norm(tokens)


# ================================================================
#  2. MoELayer — Expert-Choice top-2, 无辅助损失负载均衡
# ================================================================
class MoELayer(nn.Module):
    """MoE with Expert-Choice Routing (Auxiliary-Loss-Free)。

    原理:
      - 每个 expert 从所有 token 中选 top-k 最相关的
      - 天然负载均衡: 每个 expert 选固定数量 token, 无需 auxiliary loss
      - 被多个 expert 选中的 token 用 softmax gating 加权聚合

    对比 hash-based:
      - Hash-based: token → expert 硬编码, 忽略内容差异
      - Expert-choice: expert → token 选择, 基于内容匹配度
    """

    def __init__(self, d_model: int = 128, d_ffn: int = 512,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model),
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        flat = x.reshape(-1, D)
        N = flat.shape[0]

        gate_logits = self.gate(flat)
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Expert-Choice: 每个 expert 选 top-k token
        # tokens_per_expert = ceil(N / E) * top_k, 但为简化用固定值
        tokens_per_expert = max(1, (N * self.top_k) // self.num_experts)

        out = torch.zeros_like(flat)
        expert_load = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)

        for e in range(self.num_experts):
            # 每个 expert 选 gate_scores 最高的 tokens_per_expert 个 token
            expert_scores = gate_scores[:, e]
            top_indices = expert_scores.topk(
                min(tokens_per_expert, N), dim=0
            ).indices

            expert_load[e] = len(top_indices)

            if len(top_indices) == 0:
                continue

            expert_input = flat[top_indices]
            expert_out = self.experts[e](expert_input)

            # Gating weight: softmax over selected experts for each token
            # 简化: 用该 expert 的 gate score 作为权重
            weights = F.softmax(gate_scores[top_indices, :], dim=-1)[:, e]
            out[top_indices] += weights.unsqueeze(-1) * expert_out

        return self.norm(out.reshape(B, S, D))


# ================================================================
#  2b. TaskAwareMoELayer — 任务感知 MoE (Q专家 vs NSP专家)
# ================================================================
class TaskAwareMoELayer(nn.Module):
    """MoE with Expert-Choice Routing + 任务感知路由。

    在标准 MoE 基础上增加:
      - task_embedding: 可学习的 2 维向量 (0=Q任务, 1=NSP任务)
      - routing_bias: 任务向量 → expert 偏置 (2, E)
      - Q 任务: 前 E/2 个专家 bias ↑, 后 E/2 个专家 bias ↓
      - NSP 任务: 后 E/2 个专家 bias ↑, 前 E/2 个专家 bias ↓

    效果: 不同任务的 token 自然被路由到不同专家子集,
          无需 auxiliary loss, 无需硬编码分区。
    """

    def __init__(self, d_model: int = 128, d_ffn: int = 512,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 前半: Q-value 专家, 后半: NSP dynamics 专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model),
            )
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # 任务感知: 2 个任务 → expert 偏置
        self.task_embedding = nn.Embedding(2, num_experts)
        self.task_bias_scale = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x: torch.Tensor, task: int = 0) -> torch.Tensor:
        B, S, D = x.shape
        flat = x.reshape(-1, D)

        gate_logits = self.gate(flat)

        # 任务偏置: task=0 → Q路由, task=1 → NSP路由
        task_bias = self.task_embedding(
            torch.tensor(task, device=x.device, dtype=torch.long)
        ) * self.task_bias_scale
        gate_logits = gate_logits + task_bias.unsqueeze(0)

        gate_scores = F.softmax(gate_logits, dim=-1)

        tokens_per_expert = max(1, (flat.shape[0] * self.top_k) // self.num_experts)

        out = torch.zeros_like(flat)

        for e in range(self.num_experts):
            expert_scores = gate_scores[:, e]
            top_indices = expert_scores.topk(
                min(tokens_per_expert, flat.shape[0]), dim=0
            ).indices

            if len(top_indices) == 0:
                continue

            expert_input = flat[top_indices]
            expert_out = self.experts[e](expert_input)

            weights = F.softmax(gate_scores[top_indices, :], dim=-1)[:, e]
            out[top_indices] += weights.unsqueeze(-1) * expert_out

        return self.norm(out.reshape(B, S, D))


# ================================================================
#  3. LoopedTransformerBlock — 单个Transformer块
# ================================================================
class LoopedTransformerBlock(nn.Module):
    """单个Transformer块: Self-Attention + MoE-FFN。

    被 LoopedTransformer 重复使用 T 次 (共享参数)。
    """

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 d_ffn: int = 512, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ffn, num_experts, top_k)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.dropout(attn_out)

        x = x + self.moe(self.norm2(x))
        return x


# ================================================================
#  4. LoopedTransformer — T次循环复用同一块
# ================================================================
class LoopedTransformer(nn.Module):
    """Looped Transformer: 复用同一Transformer块 T 次。

    对比标准 Transformer:
      - 标准 L 层: L × 不同参数, O(L × d²) 参数
      - Looped T 次: 1 × 共享参数, O(d²) 参数, 深度 T 倍

    循环步编码:
      - 仅加到 CLS token 上 (修正方案)
      - CLS 作为"步间通信通道", 逐步积累全局信息
    """

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 d_ffn: int = 512, num_experts: int = 8,
                 top_k: int = 2, num_loops: int = 6,
                 state_dim: int = 268):
        super().__init__()
        self.tokenizer = InputTokenizer(d_model, state_dim)
        self.block = LoopedTransformerBlock(d_model, nhead, d_ffn, num_experts, top_k)
        self.loop_embed = nn.Embedding(num_loops, d_model)
        self.num_loops = num_loops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for t in range(self.num_loops):
            loop_pos = self.loop_embed(
                torch.tensor(t, device=x.device, dtype=torch.long)
            )
            tokens[:, 0] = tokens[:, 0] + loop_pos
            tokens = self.block(tokens)
        return tokens


# ================================================================
#  4b. TaskAwareLoopedTransformerBlock — 任务感知 Transformer 块
# ================================================================
class TaskAwareLoopedTransformerBlock(nn.Module):
    """任务感知 Transformer 块: Self-Attn + TaskAwareMoE-FFN。

    与 LoopedTransformerBlock 的区别:
      - MoE 层支持 task 参数, 不同任务路由到不同专家子集
    """

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 d_ffn: int = 512, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = TaskAwareMoELayer(d_model, d_ffn, num_experts, top_k)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, task: int = 0) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.dropout(attn_out)
        x = x + self.moe(self.norm2(x), task=task)
        return x


# ================================================================
#  4c. TaskAwareLoopedTransformer — 任务感知 Looped Transformer
# ================================================================
class TaskAwareLoopedTransformer(nn.Module):
    """任务感知 Looped Transformer: 支持 task 参数的循环复用。

    与 LoopedTransformer 的区别:
      - 使用 TaskAwareLoopedTransformerBlock (任务感知 MoE)
      - forward(x, task) 中 task 控制专家路由偏好
    """

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 d_ffn: int = 512, num_experts: int = 8,
                 top_k: int = 2, num_loops: int = 6,
                 state_dim: int = 268):
        super().__init__()
        self.tokenizer = InputTokenizer(d_model, state_dim)
        self.block = TaskAwareLoopedTransformerBlock(d_model, nhead, d_ffn, num_experts, top_k)
        self.loop_embed = nn.Embedding(num_loops, d_model)
        self.num_loops = num_loops

    def forward(self, x: torch.Tensor, task: int = 0) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for t in range(self.num_loops):
            loop_pos = self.loop_embed(
                torch.tensor(t, device=x.device, dtype=torch.long)
            )
            tokens[:, 0] = tokens[:, 0] + loop_pos
            tokens = self.block(tokens, task=task)
        return tokens


# ================================================================
#  5. EngramConditionalMemory — 条件记忆模块 (Agent层面使用)
# ================================================================
class EngramConditionalMemory(nn.Module):
    """Engram Conditional Memory — 解决幻觉问题。

    核心机制:
      - 8 个可学习记忆 slot (key, value, importance)
      - read(): 低置信度时检索记忆, 增强表征
      - write(): 高置信度时写入记忆, 积累经验
      - 幻觉抑制: 低置信度 + 记忆无匹配 → 输出低置信度

    使用方式 (在 AgentRL / MetaAgent 中):
      - self.engram = EngramConditionalMemory(d_model=128, input_dim=268)
      - 推理时: cls_output = transformer(x)[:, 0]
      - enhanced = self.engram.read(cls_output, x, confidence)
      - 训练时: self.engram.write(cls_output, x, confidence)
    """

    NUM_SLOTS = 8

    def __init__(self, d_model: int = 128, input_dim: int = 268):
        super().__init__()
        self.d_model = d_model

        # Key projection: 原始输入 → key space
        self.key_proj = nn.Linear(input_dim, d_model)
        # Query projection: CLS hidden → query space
        self.query_proj = nn.Linear(d_model, d_model)
        # Value projection: CLS hidden → value space
        self.value_proj = nn.Linear(d_model, d_model)
        # Output projection: concat(enhanced, original) → d_model
        self.output_proj = nn.Linear(d_model * 2, d_model)
        # Confidence gate: sigmoid → 0~1
        self.confidence_gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.Sigmoid(),
        )

        # 可学习记忆 slots
        self.memory_keys = nn.Parameter(torch.randn(self.NUM_SLOTS, d_model) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(self.NUM_SLOTS, d_model) * 0.02)
        self.memory_importance = nn.Parameter(torch.zeros(self.NUM_SLOTS))

    def read(self, hidden: torch.Tensor, raw_input: torch.Tensor,
             confidence: torch.Tensor) -> torch.Tensor:
        """条件读取: 置信度低时更多依赖记忆。

        Args:
            hidden: Transformer CLS输出 (B, d)
            raw_input: 原始268维输入 (B, 268)
            confidence: 当前预测置信度 (B, 1)

        Returns:
            enhanced: 增强后的表征 (B, d)
        """
        query = self.query_proj(hidden)
        B = hidden.size(0)
        keys = self.memory_keys.unsqueeze(0).expand(B, -1, -1)

        attn = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        attn = attn / math.sqrt(query.shape[-1])
        attn = F.softmax(attn, dim=-1)

        values = self.memory_values.unsqueeze(0).expand(B, -1, -1)
        memory_out = torch.bmm(attn, values).squeeze(1)

        # 置信度门控: 低置信度 → 更多依赖记忆
        gate_input = torch.cat([hidden, confidence], dim=-1)
        gate = self.confidence_gate(gate_input)

        enhanced = gate * memory_out + (1 - gate) * hidden
        return self.output_proj(torch.cat([enhanced, hidden], dim=-1))

    def write(self, hidden: torch.Tensor, raw_input: torch.Tensor,
              confidence: torch.Tensor, lr: float = 0.01):
        """条件写入: 仅在高置信度时写入记忆。

        使用 FIFO + importance-based eviction 策略。
        """
        mean_conf = confidence.mean().item()
        if mean_conf < 0.7:
            return

        key = self.key_proj(raw_input).mean(dim=0)
        value = self.value_proj(hidden).mean(dim=0)

        with torch.no_grad():
            min_imp_idx = self.memory_importance.argmin()

        self.memory_keys.data[min_imp_idx] = (
            (1 - lr) * self.memory_keys.data[min_imp_idx] + lr * key
        )
        self.memory_values.data[min_imp_idx] = (
            (1 - lr) * self.memory_values.data[min_imp_idx] + lr * value
        )
        self.memory_importance.data[min_imp_idx] = mean_conf

    def reset(self):
        """清空记忆。"""
        with torch.no_grad():
            self.memory_keys.data = torch.randn_like(self.memory_keys) * 0.02
            self.memory_values.data = torch.randn_like(self.memory_values) * 0.02
            self.memory_importance.data = torch.zeros_like(self.memory_importance)


# ================================================================
#  6. NSPHead — Next State Prediction 辅助任务头（已弃用，勿删）
# ================================================================
class NSPHead(nn.Module):
    """Next State Prediction 头 — 预测环境动态。

    .. deprecated::
        训练侧已改用 ``nsp_latent.NSPLatentModule``（EMA 目标 + InfoNCE +
        随机转移头 + H 步 rollout + 想象 Q）。本类保留是为了让旧 checkpoint
        里 ``q_network.nsp_head.*`` 的 key 仍能 strict 加载；
        ``train_step`` / ``_nsp_update`` 已不再调用它。
        删除本类会让所有既有 ``load_state_dict`` 失败（strict=True），
        因此迁移完成前不可移除。

    设计:
      - 输入: 当前状态嵌入 (d_model) + 动作嵌入 (d_action → d_model)
      - 输出: 预测的下一状态嵌入 (d_model)
      - 损失: MSE(predicted_next_state, actual_next_state)

    作用:
      - 学习环境动态: s' = f(s, a)
      - 为 Q 网络提供辅助信号, 帮助学习更好的状态表征
      - 不同于 Q 值预测, NSP 关注"发生了什么"而不是"有多好"

    与 MoE 结合:
      - NSP 专家 (后半 E/2) 专门处理动态预测
      - Q 专家 (前半 E/2) 专门处理价值评估
      - 门控网络根据当前状态动态决定激活哪些专家
    """

    def __init__(self, d_model: int = 128, num_actions: int = 7,
                 hidden_dim: int = 128):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state_embed: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """预测下一状态嵌入。

        Args:
            state_embed: 当前状态的 Transformer CLS 输出 (B, d_model)
            action: 当前动作索引 (B,) 或 one-hot (B, num_actions)

        Returns:
            predicted_next_state: 预测的下一状态嵌入 (B, d_model)
        """
        if action.dim() == 1:
            action_emb = self.action_embed(action)
        else:
            # one-hot → 嵌入
            action_emb = action @ self.action_embed.weight

        combined = torch.cat([state_embed, action_emb], dim=-1)
        return self.norm(self.mlp(combined))


# ================================================================
#  7. 单Agent Q-Network 变体
# ================================================================
class TransformerDuelingQNetwork(nn.Module):
    """Looped Transformer + MoE → Dueling Q-network。

    替换原 DuelingQNetwork (MLP)。
    接口完全兼容: forward(x) → Q-values (B, num_actions)
    """

    def __init__(self, state_dim: int = 268, d_model: int = 128,
                 nhead: int = 4, d_ffn: int = 512, num_experts: int = 8,
                 top_k: int = 2, num_loops: int = 6, num_actions: int = 7):
        super().__init__()
        self.transformer = LoopedTransformer(
            d_model, nhead, d_ffn, num_experts, top_k, num_loops, state_dim
        )
        self.pool = nn.Linear(d_model, d_model)

        self.value_stream = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.transformer(x)
        cls = tokens[:, 0]
        cls = self.pool(cls)

        val = self.value_stream(cls)
        adv = self.advantage_stream(cls)
        return val + adv - adv.mean(dim=-1, keepdim=True)


class TransformerDuelingQNetworkWithNSP(nn.Module):
    """Looped Transformer + TaskAware MoE + NSP → Dueling Q-network。

    在 TransformerDuelingQNetwork 基础上增加:
      - NSPHead: Next State Prediction 辅助任务
      - TaskAwareMoELayer: Q专家 vs NSP专家分工
      - compute_nsp_loss(): 计算 NSP 损失

    接口兼容:
      - forward(x) → Q-values (B, num_actions) — 推理时
      - forward_with_nsp(x, action, next_state_embed) → (Q, nsp_pred) — 训练时
      - compute_nsp_loss(nsp_pred, target_next_state_embed) → scalar loss
    """

    def __init__(self, state_dim: int = 268, d_model: int = 128,
                 nhead: int = 4, d_ffn: int = 512, num_experts: int = 8,
                 top_k: int = 2, num_loops: int = 6, num_actions: int = 7):
        super().__init__()
        self.transformer = TaskAwareLoopedTransformer(
            d_model, nhead, d_ffn, num_experts, top_k, num_loops, state_dim
        )
        self.pool = nn.Linear(d_model, d_model)

        self.value_stream = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_actions),
        )

        # NSP 辅助任务头
        self.nsp_head = NSPHead(d_model, num_actions, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """推理时: 只输出 Q-values。"""
        tokens = self.transformer(x, task=0)
        cls = tokens[:, 0]
        cls = self.pool(cls)
        val = self.value_stream(cls)
        adv = self.advantage_stream(cls)
        return val + adv - adv.mean(dim=-1, keepdim=True)

    def forward_with_nsp(self, x: torch.Tensor, action: torch.Tensor):
        """训练时: 输出 Q-values + NSP 预测。

        Args:
            x: 当前状态 (B, state_dim)
            action: 当前动作 (B,)

        Returns:
            q_values: Q-values (B, num_actions)
            nsp_pred: 预测的下一状态嵌入 (B, d_model)
        """
        # Q-value 分支 (task=0 → Q专家)
        tokens_q = self.transformer(x, task=0)
        cls_q = self.pool(tokens_q[:, 0])
        val = self.value_stream(cls_q)
        adv = self.advantage_stream(cls_q)
        q_values = val + adv - adv.mean(dim=-1, keepdim=True)

        # NSP 分支 (task=1 → NSP专家)
        tokens_nsp = self.transformer(x, task=1)
        cls_nsp = self.pool(tokens_nsp[:, 0])
        nsp_pred = self.nsp_head(cls_nsp, action)

        return q_values, nsp_pred

    def compute_nsp_loss(self, nsp_pred: torch.Tensor,
                         target_next_state: torch.Tensor) -> torch.Tensor:
        """计算 NSP 损失 (MSE)。

        Args:
            nsp_pred: 预测的下一状态嵌入 (B, d_model)
            target_next_state: 实际的下一状态嵌入 (B, d_model)

        Returns:
            nsp_loss: 标量损失
        """
        return F.mse_loss(nsp_pred, target_next_state.detach())


class TransformerClassificationHead(nn.Module):
    """Transformer 分类头, 替换原 ClassificationHead (MLP)。

    接口完全兼容: forward(x) → logits (B, num_classes)
    """

    def __init__(self, state_dim: int = 268, d_model: int = 128,
                 num_classes: int = 7, num_loops: int = 4):
        super().__init__()
        self.transformer = LoopedTransformer(
            d_model, num_loops=num_loops, state_dim=state_dim
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.transformer(x)
        return self.head(tokens[:, 0])


# ================================================================
#  7. 多Agent Q-Network 变体
# ================================================================
class TransformerIntentQNetwork(nn.Module):
    """多agent主意图 Q-network, 替换原 IntentQNetwork (MLP)。"""

    def __init__(self, state_dim: int = 268, d_model: int = 128,
                 num_intents: int = 7, num_loops: int = 4):
        super().__init__()
        self.transformer = LoopedTransformer(
            d_model, num_loops=num_loops, state_dim=state_dim
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_intents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.transformer(x)
        return self.head(tokens[:, 0])


class TransformerSubIntentQNetwork(nn.Module):
    """多agent子意图 Q-network, 替换原 SubIntentQNetwork (MLP)。"""

    def __init__(self, state_dim: int = 268, d_model: int = 128,
                 num_sub_intents: int = 9, num_loops: int = 4):
        super().__init__()
        self.transformer = LoopedTransformer(
            d_model, num_loops=num_loops, state_dim=state_dim
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_sub_intents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.transformer(x)
        return self.head(tokens[:, 0])


class TransformerSlotExtractionNetwork(nn.Module):
    """多任务槽位提取, 替换原 SlotExtractionNetwork (MLP)。"""

    def __init__(self, state_dim: int = 268, d_model: int = 128,
                 num_loops: int = 4):
        super().__init__()
        self.transformer = LoopedTransformer(
            d_model, num_loops=num_loops, state_dim=state_dim
        )
        self.file_head = nn.Linear(d_model, 2)
        self.image_head = nn.Linear(d_model, 2)
        self.lang_head = nn.Linear(d_model, 7)
        self.time_head = nn.Linear(d_model, 6)
        self.format_head = nn.Linear(d_model, 5)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.transformer(x)
        cls = tokens[:, 0]
        return {
            "file": F.softmax(self.file_head(cls), dim=-1),
            "image": F.softmax(self.image_head(cls), dim=-1),
            "language": F.softmax(self.lang_head(cls), dim=-1),
            "time": F.softmax(self.time_head(cls), dim=-1),
            "format": F.softmax(self.format_head(cls), dim=-1),
        }


# ================================================================
#  验证入口
# ================================================================
if __name__ == "__main__":
    print("=== Transformer Architecture Module Test ===\n")

    B = 4
    dummy_input = torch.randn(B, 268)

    # Test InputTokenizer
    tok = InputTokenizer(d_model=128)
    tokens = tok(dummy_input)
    print(f"InputTokenizer: {dummy_input.shape} → {tokens.shape}")
    assert tokens.shape == (B, 36, 128)

    # Test MoELayer
    moe = MoELayer(d_model=128, d_ffn=512, num_experts=8, top_k=2)
    moe_out = moe(tokens)
    print(f"MoELayer: {tokens.shape} → {moe_out.shape}")
    assert moe_out.shape == tokens.shape

    # Test LoopedTransformer
    lt = LoopedTransformer(d_model=128, nhead=4, d_ffn=512,
                           num_experts=8, num_loops=6, state_dim=268)
    lt_out = lt(dummy_input)
    print(f"LoopedTransformer: {dummy_input.shape} → {lt_out.shape}")
    assert lt_out.shape == (B, 36, 128)

    # Test TransformerDuelingQNetwork
    qnet = TransformerDuelingQNetwork(state_dim=268, d_model=128,
                                      num_experts=8, num_loops=6, num_actions=7)
    q_out = qnet(dummy_input)
    print(f"TransformerDuelingQNetwork: {dummy_input.shape} → {q_out.shape}")
    assert q_out.shape == (B, 7)

    # Test TransformerClassificationHead
    cls_head = TransformerClassificationHead(state_dim=268, d_model=128,
                                             num_classes=7, num_loops=4)
    cls_out = cls_head(dummy_input)
    print(f"TransformerClassificationHead: {dummy_input.shape} → {cls_out.shape}")
    assert cls_out.shape == (B, 7)

    # Test TransformerIntentQNetwork
    intent_net = TransformerIntentQNetwork(state_dim=268, d_model=128,
                                           num_intents=10, num_loops=4)
    intent_out = intent_net(dummy_input)
    print(f"TransformerIntentQNetwork: {dummy_input.shape} → {intent_out.shape}")
    assert intent_out.shape == (B, 10)

    # Test TransformerSubIntentQNetwork
    sub_net = TransformerSubIntentQNetwork(state_dim=268, d_model=128,
                                           num_sub_intents=12, num_loops=4)
    sub_out = sub_net(dummy_input)
    print(f"TransformerSubIntentQNetwork: {dummy_input.shape} → {sub_out.shape}")
    assert sub_out.shape == (B, 12)

    # Test TransformerSlotExtractionNetwork
    slot_net = TransformerSlotExtractionNetwork(state_dim=268, d_model=128,
                                                num_loops=4)
    slot_out = slot_net(dummy_input)
    print(f"TransformerSlotExtractionNetwork: {dummy_input.shape} → {{")
    for k, v in slot_out.items():
        print(f"    {k}: {v.shape}")
    assert slot_out["file"].shape == (B, 2)
    assert slot_out["language"].shape == (B, 7)

    # Test EngramConditionalMemory
    eng = EngramConditionalMemory(d_model=128, input_dim=268)
    hidden = torch.randn(B, 128)
    conf = torch.ones(B, 1) * 0.5
    eng_out = eng.read(hidden, dummy_input, conf)
    print(f"EngramConditionalMemory.read: hidden({hidden.shape}) → {eng_out.shape}")
    assert eng_out.shape == (B, 128)

    eng.write(hidden, dummy_input, torch.ones(B, 1) * 0.9)
    print(f"EngramConditionalMemory.write: done (conf=0.9)")

    # 参数量统计
    total_params = sum(p.numel() for p in qnet.parameters())
    print(f"\nTransformerDuelingQNetwork total params: {total_params:,}")
    total_params_moe = sum(p.numel() for p in qnet.transformer.block.moe.parameters())
    print(f"  MoE params: {total_params_moe:,}")

    # Test TaskAwareMoELayer
    print("\n--- TaskAwareMoE Tests ---")
    ta_moe = TaskAwareMoELayer(d_model=128, d_ffn=512, num_experts=8, top_k=2)
    ta_out_q = ta_moe(tokens, task=0)
    ta_out_nsp = ta_moe(tokens, task=1)
    print(f"TaskAwareMoE(task=0): {tokens.shape} → {ta_out_q.shape}")
    print(f"TaskAwareMoE(task=1): {tokens.shape} → {ta_out_nsp.shape}")
    assert ta_out_q.shape == tokens.shape
    assert ta_out_nsp.shape == tokens.shape

    # Test NSPHead
    nsp = NSPHead(d_model=128, num_actions=7)
    state_emb = torch.randn(B, 128)
    actions = torch.randint(0, 7, (B,))
    nsp_out = nsp(state_emb, actions)
    print(f"NSPHead: state({state_emb.shape}) + action({actions.shape}) → {nsp_out.shape}")
    assert nsp_out.shape == (B, 128)

    # Test TransformerDuelingQNetworkWithNSP
    qnet_nsp = TransformerDuelingQNetworkWithNSP(
        state_dim=268, d_model=128, num_experts=8, num_loops=6, num_actions=7
    )
    q_out_nsp = qnet_nsp(dummy_input)
    print(f"TransformerDuelingQNetworkWithNSP: {dummy_input.shape} → {q_out_nsp.shape}")
    assert q_out_nsp.shape == (B, 7)

    # Test forward_with_nsp
    actions_train = torch.randint(0, 7, (B,))
    q_vals, nsp_pred = qnet_nsp.forward_with_nsp(dummy_input, actions_train)
    print(f"  forward_with_nsp: Q={q_vals.shape}, NSP={nsp_pred.shape}")
    assert q_vals.shape == (B, 7)
    assert nsp_pred.shape == (B, 128)

    # Test compute_nsp_loss
    target_next = torch.randn(B, 128)
    nsp_loss = qnet_nsp.compute_nsp_loss(nsp_pred, target_next)
    print(f"  nsp_loss: {nsp_loss.item():.4f}")

    nsp_params = sum(p.numel() for p in qnet_nsp.parameters())
    nsp_only = sum(p.numel() for p in qnet_nsp.nsp_head.parameters())
    print(f"TransformerDuelingQNetworkWithNSP total params: {nsp_params:,}")
    print(f"  NSP head params: {nsp_only:,}")

    print("\n=== All tests passed! ===")
