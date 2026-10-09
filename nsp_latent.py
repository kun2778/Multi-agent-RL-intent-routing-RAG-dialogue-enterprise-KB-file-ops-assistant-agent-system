"""NSP 潜空间建模 —— 单 Agent 与多 Agent 共用的单副本实现。

设计动机（评审指出的四个缺口，这里各给一个落点）：

1. **MARL 侧没有 NSP** —— ``marl_agent.PolicyAgent`` 用 ``build_nsp`` 在这份
   代码上自建一个实例挂在三专家网络上，零复制
   （duplicate_gate 按函数相似度 ≥0.82 拦截，把 NSP 写两份必然撞线）。
2. **目标是"自己预测自己"** —— 旧实现拿同一网络 ``task=1`` 分支的输出当目标，
   无 target 网络，容易塌缩或追自己的尾巴。这里换成：
   - ``EMATargetEncoder``：在线编码器的 EMA 冻结副本，stop-gradient 目标；
   - InfoNCE：batch 内负样本显式防塌缩，附带 top-1 命中率作为可观测指标。
3. **只预测一步、只当 0.1 权重的正则** —— ``imagine_q_target`` 把**预测**出来的
   下一状态潜变量过 ``LatentValueBridge`` 得到想象 Q，按 α 混进 DQN 的 TD 目标；
   ``segment_losses`` 在有真实 (H+1) 步轨迹时做自由运行 H 步 rollout 监督。
4. **确定性 MSE 点估计** —— ``StochasticTransitionHead`` 输出 (μ, logσ²)，
   用高斯 NLL + KL(N(0,1)) 训练，从分布里采样而非取点估计。

指标（问题 5）统一走 ``NSPMetrics`` / ``probe()``：固定输入 → 固定输出的质量
快照，供 regression_gate、MCP 量化工具与消融脚本读取。
"""

from __future__ import annotations

import copy
import inspect
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_HORIZON = 3
LOGVAR_MIN = -10.0
LOGVAR_MAX = 6.0

# 消融档位（``apply_nsp_ablation`` 的取值域）：
#   none         全部默认，等于没消融
#   off          整块关掉（无 NLL/KL/InfoNCE、无 H 步 rollout、想象分量失效）
#   no_ema       目标副本每步直接跟在线编码器走平（退回"自己预测自己"）
#   no_infonce   去掉对比项，只剩高斯 NLL——用来看塌缩是否真的发生
#   no_bridge    去掉价值桥标定，桥只剩想象用途
#   no_hstep     H 退回 1，只监督一步（退回改造前的"一步预测"形态）
#   no_imagine   想象分量权重置 0，NSP 退化成纯正则项
#   deterministic 回到确定性点估计（MSE + 不采样），关掉随机 (μ,σ²)
ABLATION_SPECS = (
    "none", "off", "no_ema", "no_infonce", "no_bridge",
    "no_hstep", "no_imagine", "deterministic",
)


# ================================================================
#  编码器适配 —— 单 Agent 是 TaskAwareLoopedTransformer(task=)，
#  多 Agent 三专家是普通 LoopedTransformer，两者统一走这里。
# ================================================================
def encoder_supports_task(encoder: nn.Module) -> bool:
    """编码器 forward 是否接受 ``task`` 参数（任务感知路由开关）。"""
    try:
        params = inspect.signature(encoder.forward).parameters
    except (TypeError, ValueError):
        return False
    return "task" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def encode_latent(encoder: nn.Module, pool: Optional[nn.Module],
                  x: torch.Tensor, task: Optional[int] = None) -> torch.Tensor:
    """状态 → 潜向量 (B, d_model)。

    ``pool`` 为 None 时直接取 CLS（多 Agent 专家网络没有独立 pool 层）。
    """
    if task is not None and encoder_supports_task(encoder):
        tokens = encoder(x, task=task)
    else:
        tokens = encoder(x)
    cls = tokens[:, 0]
    return pool(cls) if pool is not None else cls


def discretize_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    """把（可能 multi-discrete 的）动作张量压成 [0, action_dim) 的整数码。

    SlotPolicy 的动作是 (B, 5) 五头拼接，转移头只需要一个动作 id；
    用线性同余式混合各头，对同一动作序列恒定且确定性。
    """
    dim = max(int(action_dim), 1)
    if actions.dim() == 1:
        return actions.long() % dim
    code = actions[:, 0].long()
    for col in range(1, actions.size(1)):
        code = code * 17 + actions[:, col].long()
    return code % dim


# ================================================================
#  1. StochasticTransitionHead — (z, a) → (μ, logσ²)
# ================================================================
class StochasticTransitionHead(nn.Module):
    """潜空间随机转移头：z' = f(z, a) 的对角高斯参数化。

    末层零初始化 → 初始 μ≈0、logvar≈0，KL 与 NLL 起点温和，
    不会一上来把潜空间冲散。
    """

    def __init__(self, d_model: int, action_dim: int,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or d_model * 2
        self.action_embed = nn.Embedding(max(int(action_dim), 1), d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model * 2),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor, action: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_emb = self.action_embed(
            discretize_actions(action, self.action_embed.num_embeddings))
        mu, logvar = self.net(torch.cat([z, a_emb], dim=-1)).chunk(2, dim=-1)
        return mu, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor,
               deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


# ================================================================
#  2. LatentProjector / LatentValueBridge
# ================================================================
class LatentProjector(nn.Module):
    """潜向量 → L2 归一化投影，InfoNCE 的比对空间。"""

    def __init__(self, d_model: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project a latent onto the unit sphere used by InfoNCE."""
        return F.normalize(self.net(z), dim=-1)


class LatentValueBridge(nn.Module):
    """潜向量 → Q 值：把想象出来的潜变量接回价值空间。

    被两处使用：
      - 标定：``MSE(bridge(sg(ẑ)), Q_target(s'))`` 把桥钉在真实状态上；
      - 想象：``bridge(ẑ₁)`` 提供 TD 目标里的想象 Q 分量。
    """

    def __init__(self, d_model: int, num_actions: int):
        super().__init__()
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map a latent state onto an imagined Q value for every action."""
        return self.q_head(z)

    @torch.no_grad()
    def greedy_action(self, z: torch.Tensor) -> torch.Tensor:
        """Action with the highest imagined Q — the fallback rollout action."""
        return self.q_head(z).argmax(dim=-1)


# ================================================================
#  3. EMATargetEncoder — 在线编码器的 EMA 冻结副本
# ================================================================
class EMATargetEncoder(nn.Module):
    """目标编码器：在线编码器的指数滑动平均副本，只做 stop-gradient 目标。"""

    def __init__(self, encoder: nn.Module, pool: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.pool = copy.deepcopy(pool) if pool is not None else None
        self.freeze()

    @torch.no_grad()
    def freeze(self) -> None:
        params = list(self.encoder.parameters())
        if self.pool is not None:
            params += list(self.pool.parameters())
        for p in params:
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, task: Optional[int] = None) -> torch.Tensor:
        return encode_latent(self.encoder, self.pool, x, task)

    @torch.no_grad()
    def refresh_from(self, encoder: nn.Module,
                     pool: Optional[nn.Module] = None) -> None:
        """在线编码器被重新初始化后，用它整体覆盖目标副本（非 EMA）。"""
        self.encoder.load_state_dict(encoder.state_dict())
        if pool is not None and self.pool is not None:
            self.pool.load_state_dict(pool.state_dict())
        self.freeze()

    @torch.no_grad()
    def ema_from(self, encoder: nn.Module, pool: Optional[nn.Module],
                 tau: float) -> None:
        for tp, p in zip(self.encoder.parameters(), encoder.parameters()):
            tp.mul_(1.0 - tau).add_(p.detach(), alpha=tau)
        if pool is not None and self.pool is not None:
            for tp, p in zip(self.pool.parameters(), pool.parameters()):
                tp.mul_(1.0 - tau).add_(p.detach(), alpha=tau)

    @torch.no_grad()
    def drift(self, encoder: nn.Module,
              pool: Optional[nn.Module] = None) -> float:
        """相对漂移量 ‖online − target‖ / ‖online‖，EMA 收敛度的可观测代理。"""
        pairs = list(zip(self.encoder.parameters(), encoder.parameters()))
        if pool is not None and self.pool is not None:
            pairs += list(zip(self.pool.parameters(), pool.parameters()))
        num = den = 0.0
        for tp, p in pairs:
            diff = (tp - p.detach()).float()
            num += float((diff * diff).sum())
            den += float((p.detach().float() ** 2).sum())
        return math.sqrt(num) / (math.sqrt(den) + 1e-8)


# ================================================================
#  4. NSPMetrics — 滚动指标（问题 5：过去 NSP 没有任何可观测指标）
# ================================================================
class NSPMetrics:
    """指数滑动均值的指标累加器：训练热路径 O(1) 写，按需读。"""

    def __init__(self, momentum: float = 0.02):
        self.momentum = momentum
        self._vals: Dict[str, float] = {}
        self._count = 0

    def update(self, stats: Dict[str, float]) -> None:
        self._count += 1
        m = self.momentum
        for key, value in stats.items():
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v != v or v in (float("inf"), float("-inf")):
                continue
            self._vals[key] = v if key not in self._vals else (
                (1 - m) * self._vals[key] + m * v)

    def mean(self) -> Dict[str, float]:
        return {k: round(v, 6) for k, v in sorted(self._vals.items())}

    @property
    def count(self) -> int:
        """How many update() calls ran — the counter behind every EMA mean."""
        return self._count

    def reset(self) -> None:
        self._vals.clear()
        self._count = 0


# ================================================================
#  5. NSPLatentModule — 单副本组装
# ================================================================
class NSPLatentModule(nn.Module):
    """EMA + 对比 + 随机潜变量 + H 步想象 + 价值桥接，全装在这一个类里。

    调用方只做四件事::

        nsp     = build_nsp(net.transformer, net.pool, ...)
        stats   = nsp.supervise(states, actions, next_states, q_next_vec)
        target  = nsp.imagine_q_target(states, actions, next_q, alpha)
        nsp.update_ema()          # 优化器 step 之后

    ``online_encoder`` / ``online_pool`` 以**普通属性引用**持有、不注册成子模块：
    权重归宿主网络所有（checkpoint 里已有），重复注册会让优化器拿到两份同名参数。
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        online_pool: Optional[nn.Module] = None,
        action_dim: int = 11,
        d_model: int = 128,
        num_actions: int = 11,
        horizon: int = DEFAULT_HORIZON,
        ema_tau: float = 0.02,
        beta_kl: float = 1e-3,
        temperature: float = 0.1,
        infonce_weight: float = 1.0,
        bridge_weight: float = 0.5,
        proj_dim: int = 128,
        nsp_task: Optional[int] = None,
    ):
        super().__init__()
        self.action_dim = max(int(action_dim), 1)
        self.d_model = int(d_model)
        self.num_actions = int(num_actions)
        self.horizon = max(int(horizon), 1)
        self.ema_tau = float(ema_tau)
        self.beta_kl = float(beta_kl)
        self.temperature = float(temperature)
        self.infonce_weight = float(infonce_weight)
        self.bridge_weight = float(bridge_weight)
        self.nsp_task = nsp_task
        self.enabled = True
        # 点估计消融开关：True 时 supervise 退回 MSE、rollout 不再采样
        self.deterministic = False
        # 默认值快照，reset_ablation() 靠它复位（档位可来回切）
        self._ablation_defaults = {
            "ema_tau": float(ema_tau),
            "infonce_weight": float(infonce_weight),
            "bridge_weight": float(bridge_weight),
            "horizon": int(self.horizon),
        }

        # 在线编码器只作引用、必须绕开 nn.Module.__setattr__：
        # 正常赋值会把它注册成子模块，导致宿主网络的参数在优化器里出现两次
        # （PyTorch 已就此发 duplicate parameters 警告），且 state_dict 会翻倍。
        object.__setattr__(self, "_online_encoder", online_encoder)
        object.__setattr__(self, "_online_pool", online_pool)

        self.target = EMATargetEncoder(online_encoder, online_pool)
        self.transition = StochasticTransitionHead(self.d_model, self.action_dim)
        self.projector = LatentProjector(self.d_model, proj_dim)
        self.bridge = LatentValueBridge(self.d_model, self.num_actions)
        self.metrics = NSPMetrics()

    # ------------------------------------------------------ 属性
    @property
    def online_encoder(self) -> nn.Module:
        return self._online_encoder

    @property
    def online_pool(self) -> Optional[nn.Module]:
        return self._online_pool

    def bind_online(self, encoder: nn.Module,
                    pool: Optional[nn.Module] = None) -> None:
        object.__setattr__(self, "_online_encoder", encoder)
        object.__setattr__(self, "_online_pool", pool)

    def trainable_parameters(self) -> List[nn.Parameter]:
        if not self.enabled:
            return []
        return [p for p in self.parameters() if p.requires_grad]

    def disable(self) -> None:
        """消融开关：关掉后本模块不产出梯度，损失由调用方按 0 处理。"""
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True
        for name in ("transition", "projector", "bridge"):
            for p in getattr(self, name).parameters():
                p.requires_grad_(True)
        self.target.freeze()

    def reset_ablation(self) -> List[str]:
        """复位所有可消融超参到构造时的默认值，返回发生过变化的键。

        ``apply_nsp_ablation`` 每次都先调它，因此档位之间可以互相切换，
        不需要"先记住上次改了什么"。
        """
        changed: List[str] = []
        if not self.enabled:
            self.enabled = True
            changed.append("enabled")
        if self.deterministic:
            self.deterministic = False
            changed.append("deterministic")
        for name, value in self._ablation_defaults.items():
            if getattr(self, name) != value:
                setattr(self, name, value)
                changed.append(name)
        if changed:
            # 关闭态下 enable() 会把子模块 requires_grad 打开；这里保持幂等
            for pname in ("transition", "projector", "bridge"):
                for p in getattr(self, pname).parameters():
                    if not p.requires_grad:
                        p.requires_grad_(True)
            self.target.freeze()
        return changed

    # ------------------------------------------------------ 损失原语
    def _nll(self, mu: torch.Tensor, logvar: torch.Tensor,
             z_pos: torch.Tensor) -> torch.Tensor:
        """高斯负对数似然；``deterministic`` 档退化成 MSE（点估计消融）。"""
        if self.deterministic:
            return F.mse_loss(mu, z_pos)
        var = (0.5 * logvar).exp().clamp_min(1e-6)
        return 0.5 * (((z_pos - mu) / var) ** 2
                      + logvar + math.log(2 * math.pi)).mean()

    def _kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(0, σ²‖μ, 1) 的常用写法)；点估计档没有分布可言，恒 0。"""
        if self.deterministic:
            return mu.new_zeros(())
        return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()

    def _step_latent(self, mu: torch.Tensor,
                     logvar: torch.Tensor) -> torch.Tensor:
        """rollout 的下一潜变量：随机档采样，点估计档直接用 μ。"""
        if self.deterministic:
            return mu
        return self.transition.sample(mu, logvar)

    # ------------------------------------------------------ 编码
    def _resolve_task(self, task: Optional[int]) -> Optional[int]:
        return self.nsp_task if task is None else task

    def encode_online(self, x: torch.Tensor,
                      task: Optional[int] = None) -> torch.Tensor:
        return encode_latent(self._online_encoder, self._online_pool, x,
                             self._resolve_task(task))

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor,
                      task: Optional[int] = None) -> torch.Tensor:
        """目标副本前向，恒定 stop-gradient（EMA 冻结权重）。"""
        return self.target(x, self._resolve_task(task))

    # ------------------------------------------------------ EMA
    @torch.no_grad()
    def refresh_target(self) -> None:
        """宿主编码器被重新初始化后调用，让目标副本整体对齐。"""
        self.target.refresh_from(self._online_encoder, self._online_pool)

    @torch.no_grad()
    def update_ema(self, tau: Optional[float] = None) -> None:
        if not self.enabled:
            return
        self.target.ema_from(self._online_encoder, self._online_pool,
                             self.ema_tau if tau is None else float(tau))

    @torch.no_grad()
    def ema_drift(self) -> float:
        return self.target.drift(self._online_encoder, self._online_pool)

    # ------------------------------------------------------ 一步监督
    def supervise(self, states: torch.Tensor, actions: torch.Tensor,
                  next_states: torch.Tensor,
                  q_next: Optional[torch.Tensor] = None
                  ) -> Dict[str, torch.Tensor]:
        """1 步 NSP 监督：高斯 NLL + KL + InfoNCE（+ 价值桥标定）。

        Args:
            states / next_states: 当前与下一状态原始向量 (B, D)
            actions: 1 维或多头动作张量
            q_next: 目标网络在 next_states 上的 Q 值，(B, A) 或 (B,)；
                    给了就同时标定 ``bridge``
        """
        z = self.encode_online(states)
        mu, logvar = self.transition(z, actions)
        z_pos = self.encode_target(next_states)

        nll = self._nll(mu, logvar, z_pos)
        kl = self._kl(mu, logvar)
        nce, acc = self._infonce(self.projector(mu), self.projector(z_pos))

        stats: Dict[str, torch.Tensor] = {
            "nsp_nll": nll,
            "nsp_kl": kl,
            "nsp_infonce": nce,
            "nsp_infonce_acc": acc.detach(),
        }

        bridge_mse = mu.new_zeros(())
        if q_next is not None and self.enabled:
            target = q_next.detach()
            pred = self.bridge(mu.detach())
            if target.dim() == 1 or target.size(-1) != pred.size(-1):
                idx = discretize_actions(actions, self.num_actions).view(-1, 1)
                pred = pred.gather(1, idx).squeeze(1)
                target = target.gather(1, idx).squeeze(1) \
                    if target.dim() == 2 and target.size(-1) != 1 \
                    else target.reshape(-1)
            bridge_mse = F.mse_loss(pred, target)
            stats["nsp_bridge_mse"] = bridge_mse

        total = nll + self.beta_kl * kl + self.infonce_weight * nce \
            + self.bridge_weight * bridge_mse
        stats["nsp_total"] = total
        return stats

    def _infonce(self, q: torch.Tensor, k: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """batch 内负样本 InfoNCE；正样本对来自 EMA 目标（无梯度）。"""
        logits = q @ k.detach().t() / max(self.temperature, 1e-6)
        labels = torch.arange(q.size(0), device=q.device)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

    # ------------------------------------------------------ H 步
    def rollout(self, z0: torch.Tensor, actions: Optional[torch.Tensor]
                ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """自由运行 H 步：优先用轨迹里的真实动作，缺位时用桥的贪心动作。

        Args:
            z0: 起始潜向量 (B, d)
            actions: (B, H) 真实动作序列；长度不足或为 None 时自动补齐

        Returns:
            ``[(μ_k, logσ²_k), ...]``，长度 = self.horizon
        """
        steps: List[Tuple[torch.Tensor, torch.Tensor]] = []
        z = z0
        for k in range(self.horizon):
            if actions is not None and actions.dim() == 2 and k < actions.size(1):
                act = actions[:, k]
            elif actions is not None and actions.dim() == 1 and k == 0:
                act = actions
            else:
                act = self.bridge.greedy_action(z)
            mu, logvar = self.transition(z, act)
            steps.append((mu, logvar))
            z = self._step_latent(mu, logvar)
        return steps

    def segment_losses(self, states: torch.Tensor, actions: torch.Tensor,
                       next_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """真实 (H+1) 步轨迹上的自由运行多步 NSP 监督。

        Args:
            states: (B, H+1, D) 轨迹状态
            actions: (B, H) 轨迹动作
            next_states: (B, H+1, D) 各步的下一状态

        Returns:
            含 ``nsp_total`` / ``nsp_rollout_err``；轨迹不够长时返回空字典。
        """
        if states.dim() != 3 or actions.dim() != 2 or next_states.dim() != 3:
            return {}
        horizon = min(self.horizon, actions.size(1),
                      states.size(1) - 1, next_states.size(1))
        if horizon < 1:
            return {}

        z = self.encode_online(states[:, 0])
        total = z.new_zeros(())
        rollout_err = z.new_zeros(())
        for k in range(horizon):
            mu, logvar = self.transition(z, actions[:, k])
            z_pos = self.encode_target(next_states[:, k])
            nll = self._nll(mu, logvar, z_pos)
            nce, _ = self._infonce(self.projector(mu), self.projector(z_pos))
            weight = 1.0 / (k + 1)
            total = total + weight * (nll + self.infonce_weight * nce)
            rollout_err = rollout_err + weight * F.mse_loss(mu, z_pos.detach())
            z = self._step_latent(mu, logvar)

        stats = {
            "nsp_total": total,
            "nsp_rollout_err": rollout_err / horizon,
        }
        return stats

    # ------------------------------------------------------ 想象 TD
    @torch.no_grad()
    def imagine_q_target(self, states: torch.Tensor, actions: torch.Tensor,
                         q_target_next: torch.Tensor,
                         alpha: float, clamp: float = 5.0
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把**预测**的下一状态潜变量接进 DQN 的 TD 目标。

        ``target = (1-α)·Q_target(s') + α·clamp(bridge(ẑ₁) − Q_target(s'))``，
        其中 ẑ₁ = f(enc(s), a) 是模型预测的潜变量，不是 s' 的真实编码。

        Args:
            q_target_next: 目标网络在 next_states 上的取值，(B,) 或 (B, A)
            alpha: 想象分量权重（调用方按训练进度 ramp-up）
            clamp: 想象分量相对真实目标的最大偏离，防止早期桥未收敛污染 TD

        Returns:
            (混合后的下一状态价值, 想象 Q 的取值分量)，后者仅用于指标
        """
        if alpha <= 0.0 or not self.enabled:
            return q_target_next, q_target_next
        z = self.encode_online(states)
        mu, _ = self.transition(z, actions)
        q_imag = self.bridge(mu)
        if q_target_next.dim() == 1:
            idx = discretize_actions(actions, self.num_actions).view(-1, 1)
            imag_val = q_imag.gather(1, idx).squeeze(1)
            real_val = q_target_next
        else:
            imag_val = q_imag
            real_val = q_target_next
        diff = (imag_val - real_val).clamp(-clamp, clamp)
        return real_val + float(alpha) * diff, imag_val

    # ------------------------------------------------------ 探针
    @torch.no_grad()
    def probe(self, states: torch.Tensor, actions: torch.Tensor,
              next_states: torch.Tensor,
              q_next: Optional[torch.Tensor] = None,
              segment: Optional[Dict[str, torch.Tensor]] = None
              ) -> Dict[str, float]:
        """一次前向的质量快照：不写 metrics、不更新 EMA。

        regression gate 与 MCP 量化工具都走这里 —— 固定输入 → 固定输出，
        同一份权重两次调用结果一致。
        """
        was_training = self.training
        self.eval()
        # 在线编码器不在本模块的子树里（只作引用），self.eval() 管不到它的
        # dropout；rollout 里还要采样。固定 RNG 让「同一份权重 → 同一份输出」
        # 无论调用方处于什么模式都成立。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            try:
                stats = self.supervise(states, actions, next_states, q_next)
                out = {k: float(v) for k, v in stats.items()}
                if segment:
                    out.update({k: float(v)
                                for k, v in self.segment_losses(
                                    segment["states"], segment["actions"],
                                    segment["next_states"]).items()})
                out["ema_drift"] = self.ema_drift()
                out["latent_norm"] = float(
                    self.encode_online(states).norm(dim=-1).mean())
                out["target_norm"] = float(
                    self.encode_target(next_states).norm(dim=-1).mean())
                return {k: round(v, 6) for k, v in sorted(out.items())}
            finally:
                if was_training:
                    self.train()

    def metrics_snapshot(self) -> Dict[str, float]:
        """训练期滚动指标（热路径采样得到，非确定性探针）。"""
        snap = self.metrics.mean()
        snap["updates"] = float(self.metrics.count)
        snap["ema_drift"] = self.ema_drift()
        return snap


# ================================================================
#  消融接口 —— 宿主无关（AgentRL 挂 .nsp，MetaAgent 三专家各挂一份）
# ================================================================
def nsp_modules(host: object) -> List[NSPLatentModule]:
    """宿主身上的全部 NSP 实例；两种架构的取法不同，统一在这里分叉一次。"""
    single = getattr(host, "nsp", None)
    if isinstance(single, NSPLatentModule):
        return [single]
    agents = getattr(host, "_agents", None) or {}
    return [agent.nsp for agent in agents.values()
            if isinstance(getattr(agent, "nsp", None), NSPLatentModule)]


def nsp_enabled(host: object) -> bool:
    """全部 NSP 实例都开着才算开（消融档 ``off`` 会把三个都关掉）。"""
    mods = nsp_modules(host)
    return bool(mods) and all(m.enabled for m in mods)


def apply_nsp_ablation(host: object, spec: str) -> List[str]:
    """把消融档位落到宿主上，返回本次改动的键（供日志与测试断言）。

    幂等且可逆：每次先 ``reset_ablation()`` 复位，再按档位施加，
    所以 ``apply(h, "no_hstep")`` 之后 ``apply(h, "none")`` 能回到默认。
    档位含义见模块顶部的 ``ABLATION_SPECS`` 注释。
    """
    key = str(spec).strip().lower()
    if key not in ABLATION_SPECS:
        raise ValueError(f"未知 NSP 消融档位 {spec!r}，"
                         f"可选：{', '.join(ABLATION_SPECS)}")

    touched: List[str] = []
    for nsp in nsp_modules(host):
        touched += [f"nsp.{k}" for k in nsp.reset_ablation()]
        if key == "off":
            nsp.disable()
            touched.append("nsp.enabled")
        elif key == "no_ema":
            nsp.ema_tau = 1.0
            touched.append("nsp.ema_tau")
        elif key == "no_infonce":
            nsp.infonce_weight = 0.0
            touched.append("nsp.infonce_weight")
        elif key == "no_bridge":
            nsp.bridge_weight = 0.0
            touched.append("nsp.bridge_weight")
        elif key == "no_hstep":
            nsp.horizon = 1
            touched.append("nsp.horizon")
        elif key == "deterministic":
            nsp.deterministic = True
            touched.append("nsp.deterministic")

    # 想象分量权重是宿主字段（单 Agent 与 MetaAgent 都有）
    if hasattr(host, "nsp_imagine_alpha"):
        default = float(getattr(host, "nsp_imagine_alpha_default",
                                host.nsp_imagine_alpha))
        new_alpha = 0.0 if key == "no_imagine" else default
        if float(host.nsp_imagine_alpha) != new_alpha:
            host.nsp_imagine_alpha = new_alpha
            touched.append("nsp_imagine_alpha")
    host.nsp_ablate = key
    return touched


def build_nsp(online_encoder: nn.Module,
              online_pool: Optional[nn.Module] = None,
              action_dim: int = 11,
              d_model: int = 128,
              num_actions: int = 11,
              seed: int = 20260101,
              **kwargs) -> NSPLatentModule:
    """在 fork_rng 里构造，保证新增参数的初始化不扰动宿主的全局随机流。

    regression_gate / test_marl 依赖「同一份代码两次跑出同一串预测」：若 NSP
    的 Linear 初始化消耗了全局 RNG，后面构造的专家网络权重就会跟着变。
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return NSPLatentModule(online_encoder, online_pool,
                               action_dim=action_dim, d_model=d_model,
                               num_actions=num_actions, **kwargs)


__all__ = [
    "DEFAULT_HORIZON", "ABLATION_SPECS", "NSPLatentModule", "NSPMetrics",
    "StochasticTransitionHead", "LatentProjector", "LatentValueBridge",
    "EMATargetEncoder", "build_nsp", "encode_latent",
    "encoder_supports_task", "discretize_actions",
    "nsp_modules", "nsp_enabled", "apply_nsp_ablation",
]
