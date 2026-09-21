from typing import List, Tuple
from agent_state import AgentState, IntentType, QuerySubIntent, RoutingPath, RetrievalResult, SlotType
from agent_rl import AgentRL
from marl_agent import MetaAgent, create_marl_agent
from file_operators import FileOperationHandler, OperationOutcome
from fact_query import FactQueryHandler
from deep_search import DeepSearchHandler
from clarification import ClarificationHandler
from evidence_evaluator import TerminationPolicy
from file_corpus import FileCorpus
from system_file_finder import SystemFileFinder
from ocr_handler import OCRHandler, create_ocr_handler


class Router:
    """
    调度中心（菱形节点）：根据 RL 意图将对话框输入路由到对应处理器。

    分支：
      添加/删除/修改/整理 -> FileOperationHandler 真实文件操作
      事实查询          -> FactQueryHandler  单轮向量检索
      深度分析          -> DeepSearchHandler 多轮迭代检索
      模糊             -> ClarificationHandler 澄清闭环
    每个分支执行后计算奖励、写入经验池、按终止条件判定是否结束。

    支持两种Agent架构：
      - single: 单Agent架构（原版）
      - marl: 多Agent协作架构（新版，推荐）
    """

    def __init__(self, corpus: FileCorpus = None, agent_type: str = "marl"):
        """
        Initialize Router.

        Args:
            corpus: FileCorpus instance
            agent_type: "single" for original AgentRL, "marl" for Multi-Agent RL
        """
        self.corpus = corpus or FileCorpus()
        self.agent_type = agent_type

        # Initialize agent based on type
        if agent_type == "marl":
            self.agent_rl = create_marl_agent()
            print("  [Router] 使用 MARL 多Agent架构")
        else:
            self.agent_rl = AgentRL()
            print("  [Router] 使用单Agent架构")

        self.file_operator = FileOperationHandler(self.corpus, self.corpus.root)
        self.fact_handler = FactQueryHandler(self.corpus)
        self.deep_search_handler = DeepSearchHandler(self.corpus)
        self.clarification_handler = ClarificationHandler()
        self.system_finder = SystemFileFinder()
        self.ocr_handler = create_ocr_handler()
        self.termination = TerminationPolicy()
        self.evidence_evaluator = None

    def set_evidence_evaluator(self, evaluator):
        self.evidence_evaluator = evaluator

    def route(self, state: AgentState) -> AgentState:
        """主路由入口：单回合调度 + 奖励 + 终止判定。"""
        query = state.dialog_history[-1].user_query if state.dialog_history else ""
        if not query:
            state.terminate("未接收到有效输入。")
            return state

        if query.startswith("confirm "):
            op_id = query[8:].strip()
            ok, msg = self.file_operator.confirm_operation(op_id)
            state.terminate(msg)
            return state

        if query.startswith("rollback "):
            op_id = query[9:].strip()
            ok, msg = self.file_operator.rollback_operation(op_id)
            state.terminate(msg)
            return state

        if query == "pending":
            pending = self.file_operator.get_pending_operations()
            if not pending:
                state.terminate("No pending operations.")
            else:
                lines = ["Pending operations:"]
                for op in pending:
                    lines.append(f"  - {op.operation_id}: {op.operation_type.value} ({len(op.targets)} files)")
                state.terminate("\n".join(lines))
            return state

        intent, path, confidence, clarifications = self.agent_rl.process(query, state)
        state.current_intent = intent
        state.routing_path = path

        print(f"\n{'='*62}")
        print(f"  [AgentRL] 意图: {self.agent_rl.get_intent_label(intent)} / {intent.value}"
              f"  (置信度: {confidence:.4f})")
        print(f"  [Router] 路径: {path.value}    [ε] {self.agent_rl.epsilon:.4f}")
        print(f"{'='*62}\n")

        if intent == IntentType.VAGUE:
            has_useful_clarifications = clarifications and any(len(q) > 5 for q in clarifications)
            if has_useful_clarifications:
                self._route_clarification(state, clarifications)
            else:
                print("  [保底路由] VAGUE 意图，无有效澄清问题，降级为事实查询")
                state.current_intent = IntentType.QUERY
                state.routing_path = RoutingPath.SINGLE_VECTOR
                self._route_fact_query(state, query)
        elif intent in {IntentType.ADD_FILE, IntentType.DELETE_FILE,
                        IntentType.MODIFY_FILE, IntentType.REFACTOR}:
            if intent == IntentType.DELETE_FILE and self._is_search_query(query):
                print("  [安全拦截] 检测到查询类关键词，将删除意图降级为事实查询")
                state.current_intent = IntentType.QUERY
                state.routing_path = RoutingPath.SINGLE_VECTOR
                self._route_fact_query(state, query)
            else:
                self._route_file_operation(state, intent, query)
        elif intent == IntentType.QUERY:
            sub_intent = state.current_sub_intent
            if sub_intent == QuerySubIntent.QUERY_OCR:
                self._route_ocr(state, query)
            elif sub_intent == QuerySubIntent.QUERY_SYSTEM:
                self._route_system_find(state, query)
            elif sub_intent == QuerySubIntent.QUERY_METADATA:
                self._route_query_metadata(state, query)
            elif sub_intent == QuerySubIntent.QUERY_CONTENT:
                self._route_query_content(state, query)
            elif sub_intent == QuerySubIntent.QUERY_SUMMARY:
                self._route_query_summary(state, query)
            elif sub_intent == QuerySubIntent.QUERY_TRANSLATE:
                self._route_query_translate(state, query)
            elif sub_intent == QuerySubIntent.QUERY_EXTRACT:
                self._route_query_extract(state, query)
            else:
                self._route_fact_query(state, query)
        elif intent == IntentType.DEEP_SEARCH:
            self._route_deep_search(state, query)

        state.clarifications = state.clarifications or []
        if state.clarifications:
            print("\n  [主动澄清] 以下信息需要补充后重新路由：")
            for i, q in enumerate(state.clarifications, 1):
                print(f"    Q{i}: {q}")

        return state

    # ------------------------------------------------------------- 操作分支
    def _is_search_query(self, query: str) -> bool:
        """判断查询是否包含搜索/查找类关键词。"""
        q = query.lower().strip()
        search_kws = ["找", "查", "搜索", "查询", "检索", "怎么", "什么", "哪里",
                       "search", "find", "查看", "看看", "在哪", "定位"]
        delete_kws = ["删除", "移除", "清理", "删掉", "干掉", "remove", "delete"]
        has_search = any(kw in q for kw in search_kws)
        has_delete = any(kw in q for kw in delete_kws)
        return has_search and not has_delete

    def _route_file_operation(self, state: AgentState, intent: IntentType, query: str):
        print(f"\n>>> [文件操作] {self.agent_rl.get_intent_label(intent)}")
        outcome: OperationOutcome = self.file_operator.execute(intent, query, state)

        if outcome.success:
            print(f"  [✓] {outcome.message}")
        else:
            print(f"  [✗] {outcome.message}")

        reward = self.agent_rl.compute_reward(outcome.reward_key)
        done, reason, reward_key = self.termination.check(state, reward_hint=outcome.reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({outcome.reward_key})")
        print(f"  [终止判定] {reason}")

        if done:
            state.terminate(str(outcome.message))
            return
        if state.current_round >= state.max_rounds:
            state.terminate("已超过最大回合数，强制结束本任务。")

    def _route_clarification(self, state: AgentState, clarifications: List[str]):
        print(f"\n>>> [主动澄清] 检测到模糊/不完整输入，需要澄清：")
        if not clarifications:
            clarifications = self.clarification_handler.generate_clarification(
                state.dialog_history[-1].user_query if state.dialog_history else "")
        state.clarifications = clarifications[: self.clarification_handler.max_clarifications]
        state.add_dialog(
            state.dialog_history[-1].user_query if state.dialog_history else "",
            "\n".join(state.clarifications),
        )

    def resolve_clarification(self, state: AgentState, user_answer: str) -> AgentState:
        """用户补充信息后重新进入意图识别（闭环）。"""
        print(f"\n>>> [闭环] 用户补充: {user_answer}")
        state.add_dialog(user_answer, "")
        state.clarifications = []
        reward = self.agent_rl.compute_reward("clarification_resolved")
        state.add_reward(reward)
        print(f"  [RL奖励] {reward:+.1f} (clarification_resolved)")
        return self.route(state)

    # ------------------------------------------------------------- 检索分支
    def _route_fact_query(self, state: AgentState, query: str):
        if self.system_finder.matches(query):
            self._route_system_find(state, query)
            return
        print(f"\n>>> [事实查询] 单轮向量检索: {query}")
        result = self.fact_handler.retrieve(query)
        state.add_evidence(result)
        state.increment_round()

        if result.documents:
            reward_key = "query_success"
            done = True
            reason = "已检索到相关文档，可以生成答案"
        else:
            print("  [无结果] 转 Deep Search 深度检索...")
            self.agent_rl.learn(self.agent_rl.compute_reward("insufficient_evidence"), state, False)
            self._route_deep_search(state, state.dialog_history[-1].user_query if state.dialog_history else query)
            return

        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        if done:
            state.terminate()

    def _route_system_find(self, state: AgentState, query: str):
        import time
        print(f"\n>>> [系统查找] Windows 文件与程序检索: {query}")
        t0 = time.time()
        hits, kw = self.system_finder.search(query)
        elapsed = time.time() - t0
        state.increment_round()

        docs = [{
            "path": h.get("path") or h.get("display_name", ""),
            "title": h.get("display_name", ""),
            "content": f"[{h.get('source', '系统')}] {h.get('path', '')}",
            "source": h.get("source", "系统"),
            "page": 1,
            "score": min(0.95, float(h.get("_confidence", 0.0))),
            "metadata": h,
        } for h in hits]
        result = RetrievalResult(documents=docs,
                                 score=docs[0]["score"] if docs else 0.0,
                                 query_used=query, round_num=1)
        state.add_evidence(result)

        if hits:
            print(f"  [系统查找] 定位到 {len(hits)} 条记录，最高得分: {result.score:.3f}（耗时 {elapsed:.1f}s）")
            for h in hits[:5]:
                print(f"    - [{h.get('source', '系统')}] {h.get('path') or h.get('display_name', '')}")
            answer_text = self.system_finder.format_result(query, hits)
            reward_key = "query_success"
            done = True
            reason = "已在系统中定位到相关文件/程序，可以给出答案"
        else:
            print(f"  [系统查找] 未找到匹配项（耗时 {elapsed:.1f}s），关键词: {kw}")
            answer_text = self.system_finder.format_noresult(query)
            reward_key = "insufficient_evidence"
            done = True
            reason = "系统中未找到匹配项，已给出定位建议"

        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        if done:
            state.terminate(answer_text)

    def _route_deep_search(self, state: AgentState, query: str):
        print(f"\n>>> [深度检索] 多轮迭代检索: {query}")
        self.deep_search_handler.execute(state, self.termination, self)

        done, reason, reward_key = self.termination.check(state)
        
        # 检查是否有有效结果（生成了答案或报告）
        has_valid_result = self._check_valid_result(state)
        
        # 检查是否有噪声文件干扰
        has_noise = self._check_noise_interference(state)
        
        # 检查是否违反了禁止操作
        violations = self._check_violations(state)
        
        # 检查是否覆盖了关键槽位
        slots_covered = self._check_key_slots_covered(state, query)
        
        # 检查文档多样性
        file_diversity = self._check_file_diversity(state)
        
        # 奖励判定逻辑（更严格）：
        # 1. 必须有有效结果
        # 2. 不能有噪声干扰
        # 3. 不能违反禁止操作
        # 4. 必须覆盖关键槽位
        # 5. 必须有足够的文件多样性
        
        if done and reward_key == "evidence_sufficient":
            if has_valid_result and not has_noise and not violations and slots_covered and file_diversity >= 3:
                reward_key = "deep_search_success"
                reason = f"任务完成（有效结果: ✓，噪声: ✓，槽位覆盖: ✓，文件多样性: {file_diversity}）"
            else:
                # 没有通过验证，给负奖励
                reward_key = "insufficient_evidence"
                issues = []
                if not has_valid_result:
                    issues.append("无有效结果")
                if has_noise:
                    issues.append("噪声干扰")
                if violations:
                    issues.append(f"违反操作: {', '.join(violations)}")
                if not slots_covered:
                    issues.append("未覆盖关键槽位")
                if file_diversity < 3:
                    issues.append(f"文件多样性不足 ({file_diversity})")
                reason = f"任务未完成：{', '.join(issues)}"
        
        # 如果不是 evidence_sufficient，保持原样
        # 例如 timeout 或其他终止原因
        
        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        if done:
            state.terminate()
    
    def _check_valid_result(self, state: AgentState) -> bool:
        """检查深度检索是否有有效结果。"""
        # 检查是否有最终答案
        if state.final_answer:
            # 答案应该不是空的或者只是简单的模板
            if len(state.final_answer) > 50 and "未找到" not in state.final_answer:
                return True
        
        # 检查是否有高质量的检索结果
        if state.evidence:
            high_quality_docs = 0
            for ev in state.evidence:
                for doc in ev.documents:
                    score = doc.get("score", 0.0)
                    path = doc.get("path", "").lower()
                    # 排除噪声文件
                    if score >= 0.5 and not any(noise in path for noise in 
                        ['.operation_log.json', '.log', '.tmp', '.lock']):
                        high_quality_docs += 1
            
            if high_quality_docs >= 2:
                return True
        
        return False
    
    def _check_noise_interference(self, state: AgentState) -> bool:
        """检查是否有噪声文件干扰。"""
        noise_patterns = ['.operation_log.json', '.log', '.tmp', '.lock', '.min.js', '.map']
        
        for ev in state.evidence:
            for doc in ev.documents:
                path = doc.get("path", "").lower()
                if any(noise in path for noise in noise_patterns):
                    # 检查噪声文件是否占主导（得分最高）
                    score = doc.get("score", 0.0)
                    if score > 0.5:  # 噪声文件得分过高
                        return True
        
        return False
    
    def _check_violations(self, state: AgentState) -> List[str]:
        """检查是否违反了禁止操作。"""
        violations = []
        
        # 检查是否有文件被修改（通过检查 final_answer 中是否包含修改指令）
        if state.final_answer:
            answer = state.final_answer.lower()
            modification_keywords = ["已修改", "已删除", "已移动", "已重命名", "已覆盖"]
            for kw in modification_keywords:
                if kw in answer:
                    violations.append(f"检测到{kw}操作")
        
        return violations
    
    def _check_key_slots_covered(self, state: AgentState, query: str) -> bool:
        """检查是否覆盖了关键槽位。"""
        q = query.lower()
        
        # 根据查询类型检查不同的槽位
        if "todo" in q or "fixme" in q or "xxx" in q or "hack" in q:
            # 应该检索到包含这些标记的文件
            for ev in state.evidence:
                for doc in ev.documents:
                    content = doc.get("content", "").upper()
                    if any(mark in content for mark in ["TODO", "FIXME", "XXX", "HACK"]):
                        return True
            return False
        
        if "依赖" in q or "import" in q:
            # 应该检索到包含导入语句的文件
            for ev in state.evidence:
                for doc in ev.documents:
                    content = doc.get("content", "")
                    if "import" in content or "from" in content:
                        return True
            return False
        
        # 默认：至少有一些有效文档
        return len(state.evidence) > 0
    
    def _check_file_diversity(self, state: AgentState) -> int:
        """检查文件多样性（不同文件的数量）。"""
        unique_files = set()
        for ev in state.evidence:
            for doc in ev.documents:
                path = doc.get("path", "")
                if path:
                    unique_files.add(path)
        return len(unique_files)

    def _route_ocr(self, state: AgentState, query: str):
        """路由到OCR处理分支。"""
        print(f"\n>>> [OCR识别] 图片文字识别: {query}")

        ocr_result = self.ocr_handler.process_ocr_request(state, query)

        state.add_evidence(RetrievalResult(
            documents=[{"content": ocr_result, "source": "ocr", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()

        reward_key = "query_success" if "识别结果" in ocr_result else "insufficient_evidence"
        done = True
        reason = "OCR识别完成" if reward_key == "query_success" else "OCR识别失败"

        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(ocr_result)

    def _route_query_metadata(self, state: AgentState, query: str):
        """路由到文件元数据查询。"""
        print(f"\n>>> [文件信息] 查询文件元数据: {query}")

        target = state.slots.get("file_path") or self.agent_rl.query_rewriter.extract_target(query)
        if not target:
            state.clarifications = ["请指定要查询信息的文件名或路径。"]
            return

        result = self.corpus.get_file_metadata(target)
        if result:
            content = f"文件信息:\n{result}"
            reward_key = "query_success"
        else:
            content = f"未找到文件: {target}"
            reward_key = "insufficient_evidence"

        state.add_evidence(RetrievalResult(
            documents=[{"content": content, "source": "metadata", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()

        done = True
        reason = "文件信息查询完成" if reward_key == "query_success" else "未找到文件"

        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(content)

    def _route_query_content(self, state: AgentState, query: str):
        """路由到文件内容查询。"""
        print(f"\n>>> [文件内容] 读取文件内容: {query}")

        target = state.slots.get("file_path") or self.agent_rl.query_rewriter.extract_target(query)
        if not target:
            state.clarifications = ["请指定要读取内容的文件名或路径。"]
            return

        content = self.corpus.read_text(target)
        if content:
            reward_key = "query_success"
            display_content = f"[{target}] 内容:\n{content[:500]}{'...' if len(content) > 500 else ''}"
        else:
            reward_key = "insufficient_evidence"
            display_content = f"无法读取文件内容: {target}"

        state.add_evidence(RetrievalResult(
            documents=[{"content": display_content, "source": "content", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()

        done = True
        reason = "文件内容读取完成" if reward_key == "query_success" else "无法读取文件"

        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(display_content)

    def _route_query_summary(self, state: AgentState, query: str):
        """路由到内容总结。"""
        print(f"\n>>> [内容总结] 总结文件内容: {query}")

        target = state.slots.get("file_path") or self.agent_rl.query_rewriter.extract_target(query)
        if not target:
            state.clarifications = ["请指定要总结内容的文件名或路径。"]
            return

        content = self.corpus.read_text(target)
        if not content:
            state.terminate(f"无法读取文件内容: {target}")
            return

        summary = self._generate_summary(content, query)
        reward_key = "query_success"

        state.add_evidence(RetrievalResult(
            documents=[{"content": summary, "source": "summary", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()

        done = True
        reason = "内容总结完成"

        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(summary)

    def _route_query_translate(self, state: AgentState, query: str):
        """路由到内容翻译。"""
        print(f"\n>>> [内容翻译] 翻译文件内容: {query}")

        target = state.slots.get("file_path") or self.agent_rl.query_rewriter.extract_target(query)
        language = state.slots.get("language", "英文")

        if not target:
            state.clarifications = ["请指定要翻译的文件名或路径。"]
            return

        content = self.corpus.read_text(target)
        if not content:
            state.terminate(f"无法读取文件内容: {target}")
            return

        translated = self._translate_content(content, language, query)
        reward_key = "query_success"

        state.add_evidence(RetrievalResult(
            documents=[{"content": translated, "source": "translate", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()

        done = True
        reason = "内容翻译完成"

        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(translated)

    def _route_query_extract(self, state: AgentState, query: str):
        """路由到字段提取。"""
        print(f"\n>>> [字段提取] 提取文件信息: {query}")

        target = state.slots.get("file_path") or self.agent_rl.query_rewriter.extract_target(query)
        if not target:
            state.clarifications = ["请指定要提取信息的文件名或路径。"]
            return

        content = self.corpus.read_text(target)
        if not content:
            state.terminate(f"无法读取文件内容: {target}")
            return

        extracted = self._extract_information(content, query)
        reward_key = "query_success"

        state.add_evidence(RetrievalResult(
            documents=[{"content": extracted, "source": "extract", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()

        done = True
        reason = "字段提取完成"

        reward = self.agent_rl.compute_reward(reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(extracted)

    def _generate_summary(self, content: str, query: str) -> str:
        """生成内容摘要。"""
        lines = content.split('\n')
        total_lines = len(lines)
        total_chars = len(content)

        keywords = self._extract_keywords(content)

        summary = f"  [内容摘要]\n"
        summary += f"  文件统计: {total_lines} 行, {total_chars} 字符\n"
        summary += f"  关键词: {', '.join(keywords[:5])}\n\n"
        summary += f"  内容概览:\n"

        non_empty_lines = [line.strip() for line in lines if line.strip()]
        preview_lines = non_empty_lines[:10]
        for line in preview_lines:
            summary += f"    {line[:80]}{'...' if len(line) > 80 else ''}\n"

        if total_lines > 10:
            summary += f"    ... (共 {total_lines} 行)\n"

        return summary

    def _translate_content(self, content: str, target_language: str, query: str) -> str:
        """翻译内容（简单实现，实际应调用LLM）。"""
        lang_map = {
            "英文": "English", "中文": "Chinese", "日文": "Japanese",
            "韩文": "Korean", "法文": "French", "德文": "German",
            "english": "English", "chinese": "Chinese", "japanese": "Japanese",
        }
        lang_name = lang_map.get(target_language, target_language)

        preview = content[:300] + "..." if len(content) > 300 else content

        return (f"  [翻译预览]\n"
                f"  目标语言: {lang_name}\n"
                f"  原文预览:\n{preview}\n\n"
                f"  注意: 完整翻译需要配置LLM服务。当前显示原文预览。")

    def _extract_information(self, content: str, query: str) -> str:
        """从内容中提取信息。"""
        import re

        emails = re.findall(r'[\w.-]+@[\w.-]+\.\w+', content)
        urls = re.findall(r'https?://\S+', content)
        numbers = re.findall(r'\b\d+\.?\d*\b', content)

        result = f"  [信息提取结果]\n"

        if emails:
            result += f"  邮箱地址 ({len(emails)} 个):\n"
            for email in emails[:5]:
                result += f"    - {email}\n"

        if urls:
            result += f"  网址链接 ({len(urls)} 个):\n"
            for url in urls[:5]:
                result += f"    - {url}\n"

        if numbers:
            result += f"  数字信息 ({len(numbers)} 个):\n"
            for num in numbers[:10]:
                result += f"    - {num}\n"

        if not (emails or urls or numbers):
            result += "  未找到特定格式的信息。\n"
            result += f"  内容预览:\n{content[:200]}..."

        return result

    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词。"""
        import re
        from collections import Counter

        words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', content)
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(10) if len(word) > 1]

    def switch_agent(self, agent_type: str):
        """Switch between single-agent and MARL architectures at runtime.

        Args:
            agent_type: "single" or "marl"
        """
        if agent_type == self.agent_type:
            return

        if agent_type == "marl":
            self.agent_rl = create_marl_agent()
            print("  [Router] 切换到 MARL 多Agent架构")
        else:
            self.agent_rl = AgentRL()
            print("  [Router] 切换到单Agent架构")

        self.agent_type = agent_type

    def compare_agents(self, query: str) -> dict:
        """Compare predictions from both single-agent and MARL architectures.

        Useful for debugging and understanding differences.

        Args:
            query: User input query

        Returns:
            Dictionary with comparison results
        """
        from agent_state import AgentState as AS
        import copy

        results = {"query": query}

        # Test with current agent
        state1 = AS()
        state1.increment_round()
        state1.corpus_size = len(self.corpus.registry)

        # Save current agent
        original_agent = self.agent_rl
        original_type = self.agent_type

        # Test single agent
        self.agent_rl = AgentRL()
        self.agent_type = "single"
        try:
            intent1, path1, conf1, clar1 = self.agent_rl.process(query, state1)
            results["single_agent"] = {
                "intent": intent1.value,
                "path": path1.value if path1 else None,
                "confidence": conf1,
                "clarifications": clar1,
            }
        except Exception as e:
            results["single_agent"] = {"error": str(e)}

        # Test MARL agent
        self.agent_rl = create_marl_agent()
        self.agent_type = "marl"
        state2 = AS()
        state2.increment_round()
        state2.corpus_size = len(self.corpus.registry)
        try:
            intent2, path2, conf2, clar2 = self.agent_rl.process(query, state2)
            results["marl_agent"] = {
                "intent": intent2.value,
                "path": path2.value if path2 else None,
                "confidence": conf2,
                "clarifications": clar2,
            }
        except Exception as e:
            results["marl_agent"] = {"error": str(e)}

        # Restore original agent
        self.agent_rl = original_agent
        self.agent_type = original_type

        # Add comparison
        if "error" not in results.get("single_agent", {}) and "error" not in results.get("marl_agent", {}):
            results["match"] = results["single_agent"]["intent"] == results["marl_agent"]["intent"]

        return results