from typing import Tuple, List, Optional
from agent_state import AgentState, IntentType, RoutingPath


OP_INTENTS = {
    IntentType.ADD_FILE, IntentType.DELETE_FILE,
    IntentType.MODIFY_FILE, IntentType.REFACTOR,
}


class TerminationPolicy:
    """终止条件模块（双条件终止判断）。

    终止条件由三部分构成：
      1. 任务完成   ：文件操作已提交成功，或检索证据充分（子查询全部覆盖且无矛盾）
      2. 资源预算   ：检索轮次达到上限 / 时间超过预算 -> 强制终止（奖励键 timeout）
      3. 澄清闭环   ：VAGUE 意图不终止，要求用户补充信息后重新路由

    check(state, reward_hint=None) 返回:
      (should_terminate, reason, reward_key)
    reward_hint 可携带外部已确定的奖励键（如文件操作成功 add_success）。
    """

    def __init__(self, max_rounds: int = 6, time_threshold: float = 120.0):
        self.max_rounds = max_rounds
        self.time_threshold = time_threshold

    def check(self, state: AgentState, reward_hint: str = None, reward_key: str = None) -> Tuple[bool, str, str]:
        # reward_key 兼容别名
        if reward_key and not reward_hint:
            reward_hint = reward_key
        intent = state.current_intent

        # ---- 条件 1+3：操作类任务，按真实操作结果判定 ----
        if intent in OP_INTENTS:
            if reward_hint and reward_hint.endswith("_success"):
                op = state.last_operation()
                msg = op.message if op else "操作已提交"
                return True, msg, reward_hint
            # 操作失败：不直接终止，继续走预算兜底（允许下一回合重试）

        # ---- 条件 2：资源预算（对所有分支最后兜底） ----
        if state.current_round >= self.max_rounds:
            return True, f"达到最大回合数 ({self.max_rounds})，强制终止", "timeout"
        if state.elapsed_time > self.time_threshold:
            return True, f"超过时间预算 ({self.time_threshold:.0f}s)，强制终止", "timeout"

        # ---- 检索类任务：证据充分性 + 预算 ----
        if intent in (IntentType.QUERY, IntentType.DEEP_SEARCH):
            sufficient, reason = self._check_evidence(state)
            if sufficient:
                return True, reason, "evidence_sufficient"
            return False, reason, "insufficient_evidence"

        # VAGUE：不终止，等待澄清闭环
        if intent == IntentType.VAGUE:
            return False, "等待用户澄清", ""

        # 当前无明确意图（兜底）：任务进行中
        return False, "任务进行中", ""

    def evaluate(self, state: AgentState, reward_hint: str = None) -> Tuple[bool, str]:
        """兼容 evaluate() 接口（返回 (done, reason)）。"""
        done, reason, _ = self.check(state, reward_hint)
        return done, reason

    # ------------------------------------------------------------------ 证据
    def _check_evidence(self, state: AgentState) -> Tuple[bool, str]:
        """检查证据充分性，基于证据覆盖度和质量，而不是轮数。"""
        if not state.evidence:
            return False, "尚未收集到证据（无检索结果）"

        if self._detect_contradictions(state):
            return False, "证据之间存在矛盾，需要继续检索澄清"

        # 计算证据质量指标
        quality_score = self._calculate_evidence_quality(state)
        
        # 检查是否有噪声文件（如.operation_log.json）
        has_noise = self._has_noise_files(state)
        if has_noise:
            quality_score *= 0.5  # 降低噪声文件的质量分数
        
        # 检查是否覆盖了关键槽位
        slots_covered = self._check_slots_coverage(state)
        
        # 检查检索的子查询数量
        evidence_count = len(state.evidence)
        
        # 检查文档多样性（不同文件的数量）
        unique_files = set()
        for ev in state.evidence:
            for doc in ev.documents:
                path = doc.get("path", "")
                if path:
                    unique_files.add(path)
        file_diversity = len(unique_files)
        
        # 证据充分的条件（更严格）：
        # 1. 质量分数 > 0.7 且覆盖了关键槽位 且至少有3个不同文件
        # 2. 或者至少4轮检索 且质量 > 0.5 且有3个以上不同文件
        # 3. 质量分数不能低于0.8（避免0.81->0.80这种矛盾）
        
        if quality_score > 0.7 and slots_covered and file_diversity >= 3:
            return True, f"证据充分（质量: {quality_score:.2f}，{file_diversity} 个不同文件），可以生成答案"
        elif evidence_count >= 4 and quality_score > 0.5 and file_diversity >= 3:
            return True, f"证据充分（质量: {quality_score:.2f}，{evidence_count} 轮检索，{file_diversity} 个文件），可以生成答案"
        else:
            reason_parts = []
            if quality_score <= 0.7:
                reason_parts.append(f"质量不足 ({quality_score:.2f} < 0.7)")
            if not slots_covered:
                reason_parts.append("未覆盖关键槽位")
            if file_diversity < 3:
                reason_parts.append(f"文件多样性不足 ({file_diversity} < 3)")
            return False, f"证据不充分：{', '.join(reason_parts) if reason_parts else '需要更多检索'}"
    
    def _calculate_evidence_quality(self, state: AgentState) -> float:
        """计算证据质量分数。"""
        if not state.evidence:
            return 0.0
        
        total_score = 0.0
        doc_count = 0
        
        for ev in state.evidence:
            for doc in ev.documents:
                score = doc.get("score", 0.0)
                # 过滤掉低质量文档（得分 < 0.3）
                if score >= 0.3:
                    total_score += score
                    doc_count += 1
        
        if doc_count == 0:
            return 0.0
        
        # 平均得分
        avg_score = total_score / doc_count
        
        # 文档多样性奖励（不同文件的文档）
        unique_files = set()
        for ev in state.evidence:
            for doc in ev.documents:
                unique_files.add(doc.get("path", ""))
        diversity_bonus = min(len(unique_files) / 5.0, 1.0)  # 最多5个不同文件
        
        return min(avg_score * (1 + diversity_bonus * 0.3), 1.0)
    
    def _has_noise_files(self, state: AgentState) -> bool:
        """检查证据中是否包含噪声文件。"""
        noise_patterns = ['.operation_log.json', '.log', '.tmp', '.lock', '.min.js', '.map']
        for ev in state.evidence:
            for doc in ev.documents:
                path = doc.get("path", "").lower()
                if any(noise in path for noise in noise_patterns):
                    return True
        return False
    
    def _check_slots_coverage(self, state: AgentState) -> bool:
        """检查是否覆盖了关键槽位。"""
        # 关键槽位：至少有一些有效的检索结果
        if not state.evidence:
            return False
        
        # 检查是否有足够多的有效文档
        valid_doc_count = 0
        for ev in state.evidence:
            for doc in ev.documents:
                if doc.get("score", 0.0) >= 0.3:
                    valid_doc_count += 1
        
        return valid_doc_count >= 2  # 至少2个有效文档

    def _detect_contradictions(self, state: AgentState) -> bool:
        all_text = " ".join(
            doc["content"] for ev in state.evidence for doc in ev.documents
        )
        contradictory_pairs = [
            ("增长", "下降"), ("上升", "减少"), ("成功", "失败"),
            ("正面", "负面"), ("盈利", "亏损"), ("扩张", "收缩"),
        ]
        return any(pos in all_text and neg in all_text for pos, neg in contradictory_pairs)


class EvidenceEvaluator:
    """兼容旧接口的证据充分性评估（检索类任务的终止条件之一）。"""

    def __init__(self):
        self.policy = TerminationPolicy()

    def evaluate(self, state: AgentState) -> Tuple[bool, str]:
        done, reason, _reward = self.policy.check(state)
        return done, reason

    def check(self, state: AgentState, reward_hint: str = None) -> Tuple[bool, str, str]:
        return self.policy.check(state, reward_hint)