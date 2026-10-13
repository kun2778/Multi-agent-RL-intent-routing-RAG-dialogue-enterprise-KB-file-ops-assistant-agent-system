from typing import List, Tuple, Dict, Optional
import traceback
from agent_state import AgentState, IntentType, QuerySubIntent, RoutingPath, RetrievalResult, SlotType
from agent_rl import AgentRL, should_clarify, MAX_CLARIFY_DEPTH, WRITE_CONF_GUARD
from marl_agent import MetaAgent, create_marl_agent
from file_operators import FileOperationHandler, OperationOutcome
from fact_query import FactQueryHandler
from deep_search import DeepSearchHandler
from clarification import ClarificationHandler
from evidence_evaluator import TerminationPolicy
from file_corpus import FileCorpus
from system_file_finder import SystemFileFinder
from filelist_search import FileListSearchHandler
from ocr_handler import OCRHandler, create_ocr_handler
from intent_shared import SEARCH_KEYWORDS, DELETE_KEYWORDS


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
        self.filelist_handler = FileListSearchHandler()
        self.ocr_handler = create_ocr_handler()
        self.termination = TerminationPolicy()
        self.evidence_evaluator = None

    def set_evidence_evaluator(self, evaluator):
        self.evidence_evaluator = evaluator

    def _slot_file_target(self, state: AgentState, query: str) -> Optional[str]:
        """从槽位/查询中安全提取目标文件路径，过滤 present/absent 等占位值。"""
        target = state.slots.get("file_path")
        if target is not None and str(target).strip().lower() in ("present", "absent", "none", "null", ""):
            target = None
        if not target:
            target = self.agent_rl.query_rewriter.extract_target(query)
        return target

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
            # 期望决策：conf < p* 才询问，否则降级回答（牺牲奖励 vs 冒答错风险）
            want_clarify = should_clarify(confidence, state)
            has_useful_clarifications = clarifications and any(len(q) > 5 for q in clarifications)
            if want_clarify and has_useful_clarifications:
                self._route_clarification(state, clarifications)
            else:
                print(f"  [保底路由] VAGUE 意图，期望决策={want_clarify} / 有效澄清={bool(has_useful_clarifications)}，降级为事实查询")
                state.current_intent = IntentType.QUERY
                state.routing_path = RoutingPath.SINGLE_VECTOR
                self._route_fact_query(state, query)
        elif intent in {IntentType.ADD_FILE, IntentType.DELETE_FILE,
                        IntentType.MODIFY_FILE, IntentType.REFACTOR}:
            if self._is_search_query(query):
                print("  [安全拦截] 检测到查询类关键词，将操作意图降级为事实查询")
                state.current_intent = IntentType.QUERY
                state.routing_path = RoutingPath.SINGLE_VECTOR
                self._route_fact_query(state, query)
            elif confidence < WRITE_CONF_GUARD:
                # 护栏：置信度低于下限（如 LLM 缺席时神经路盲猜 0.14），
                # 意图本身不可信 → 不执行写操作，降级为事实查询保底。
                print(f"  [护栏] 写意图置信度过低 ({confidence:.4f} < {WRITE_CONF_GUARD})，"
                      f"降级为事实查询")
                state.current_intent = IntentType.QUERY
                state.routing_path = RoutingPath.SINGLE_VECTOR
                self._route_fact_query(state, query)
            elif (intent != IntentType.REFACTOR
                  and not self._slot_file_target(state, query)
                  and state.clarification_rounds < MAX_CLARIFY_DEPTH):
                # 缺关键槽位（文件名）：消费 process() 已生成的澄清信号，
                # 先问再做，不再直接执行后必然失败。
                self._route_clarification(state, clarifications)
            else:
                self._route_file_operation(state, intent, query)
        elif intent == IntentType.QUERY:
            sub_intent = state.current_sub_intent
            if sub_intent == QuerySubIntent.QUERY_OCR and not self._ocr_warranted(
                    state, query):
                print("  [OCR守卫] 未检测到图片路径/OCR语义，OCR 子意图降级为事实查询")
                state.current_sub_intent = QuerySubIntent.QUERY_OTHER
                state.routing_path = RoutingPath.SINGLE_VECTOR
                self._route_fact_query(state, query)
            elif sub_intent == QuerySubIntent.QUERY_OCR:
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
            elif sub_intent == QuerySubIntent.QUERY_FILE_SCAN:
                self._route_file_scan(state, query)
            elif sub_intent == QuerySubIntent.QUERY_FILE_ANALYZE:
                self._route_file_analysis(state, query)
            elif sub_intent == QuerySubIntent.QUERY_READONLY_REPORT:
                self._route_report_generate(state, query)
            else:
                self._route_fact_query(state, query)
        elif intent == IntentType.DEEP_SEARCH:
            self._route_deep_search(state, query)
        elif intent == IntentType.FILE_ANALYSIS:
            self._route_file_analysis(state, query)
        elif intent == IntentType.FILE_SCAN:
            self._route_file_scan(state, query)
        elif intent == IntentType.REPORT_GENERATE:
            self._route_report_generate(state, query)
        elif intent == IntentType.FILENAME_QUERY:
            self._route_filename_query(state, query)
        elif intent == IntentType.FILELIST_SEARCH:
            self._route_filelist_search(state, query)

        state.clarifications = state.clarifications or []
        if state.clarifications:
            print("\n  [主动澄清] 以下信息需要补充后重新路由：")
            for i, q in enumerate(state.clarifications, 1):
                print(f"    Q{i}: {q}")

        return state

    # ------------------------------------------------------------- 操作分支
    def _ocr_warranted(self, state: AgentState, query: str) -> bool:
        """OCR 分支必须有图像证据，否则子意图可能是误采样/误判。

        三类正证据（满足其一即可路由 OCR）：
          1) 槽位里真的有图片路径；
          2) 查询显式提到 OCR 动作词（识别/ocr/文字识别）；
          3) 查询含图像关键词（截屏/图片/...，query_rewriter.IMAGE_KEYWORDS）。
        证据缺失时返回 False，由调用方降级到事实查询，
        避免「无图 → OCR失败 → insufficient_evidence(-2.0)」的空转。
        """
        if state.slots.get(SlotType.IMAGE_PATH.value):
            return True
        q = (query or "").lower()
        if any(kw in q for kw in ("识别", "ocr", "文字识别")):
            return True
        return self.agent_rl.query_rewriter.detect_ocr_need(query)

    def _is_search_query(self, query: str) -> bool:
        """判断查询是否包含搜索/查找类关键词。

        用 intent_shared 的统一词表：这里与 agent_rl/marl 的查询判定必须同口径，
        否则「单体认为是查询、Router 认为不是」会让安全拦截失效。
        """
        q = query.lower().strip()
        has_search = any(kw in q for kw in SEARCH_KEYWORDS)
        has_delete = any(kw in q for kw in DELETE_KEYWORDS)
        return has_search and not has_delete

    def _route_file_operation(self, state: AgentState, intent: IntentType, query: str):
        print(f"\n>>> [文件操作] {self.agent_rl.get_intent_label(intent)}")
        outcome: OperationOutcome = self.file_operator.execute(intent, query, state)

        if outcome.success:
            print(f"  [✓] {outcome.message}")
        else:
            print(f"  [✗] {outcome.message}")

        reward = self.agent_rl.compute_reward(outcome.reward_key, state)
        done, reason, reward_key = self.termination.check(state, reward_hint=outcome.reward_key)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({outcome.reward_key})  [累计: {len(state.rl_rewards)} 事件]")
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
        """用户补充信息后重新进入意图识别（闭环）。

        澄清奖励在此写入经验池：transition 的 (s, a) 是上一轮 classify_intent
        选出的 VAGUE 动作，(s') 是补充后的用户输入，done=False 表示回合继续、
        bootstrap 链不被切断。必须在 route() 之前 learn——route() 内部的
        classify_intent/select_action 会覆盖 last_action 与 last_features，
        否则询问奖励会被记到新一轮的动作上。

        奖励随轮次衰减：第 k 次澄清拿到 ask_reward(k)=q_k，随后
        clarification_rounds += 1，使紧随其后的答对拿到 a_{k+1}，
        从而路径总量满足 P1 < P2 = P3 > P4 > ... > 答错。
        """
        print(f"\n>>> [闭环] 用户补充: {user_answer}")
        # 澄清答案与原指令合并后重新路由：写意图缺槽位的澄清问题只要求补
        # 文件名（如「notes.txt」），拿答案单独分类会跑偏；合并后
        # 「添加文件」+「notes.txt」= 完整指令，VAGUE 场景用户给全指令也不受影响。
        original = state.dialog_history[-1].user_query if state.dialog_history else ""
        merged = f"{original} {user_answer.strip()}".strip()
        state.add_dialog(merged, "")
        state.clarifications = []
        reward = self.agent_rl.compute_reward("clarification_resolved", state)
        self.agent_rl.learn(reward, state, done=False)
        state.clarification_rounds += 1
        print(f"  [RL奖励] {reward:+.3f} (clarification_resolved, 第 {state.clarification_rounds} 轮澄清)")
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
            self.agent_rl.learn(self.agent_rl.compute_reward("insufficient_evidence", state), state, False)
            self._route_deep_search(state, state.dialog_history[-1].user_query if state.dialog_history else query)
            return

        reward = self.agent_rl.compute_reward(reward_key, state)
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

        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        if done:
            state.terminate(answer_text)

    def _route_deep_search(self, state: AgentState, query: str):
        print(f"\n>>> [深度检索] 多轮迭代检索: {query}")
        # 深度检索内部异常不允许拖垮整条链路：记录堆栈后降级为单轮事实检索
        try:
            self.deep_search_handler.execute(state, self.termination, self)
        except Exception:
            traceback.print_exc()
            print("  [深度检索] 执行异常，已降级为单轮事实检索（本轮不中断）")
            try:
                state.add_evidence(self.fact_handler.retrieve(query))
                state.increment_round()
            except Exception:
                traceback.print_exc()

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
        
        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        if done:
            if state.final_answer:
                # ReAct 已产出答案：只标记终止，不能用空串覆盖掉它
                state.is_terminated = True
            else:
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

    def _route_file_analysis(self, state: AgentState, query: str):
        """路由到文件分析分支（只读深度分析）。"""
        print(f"\n>>> [文件分析] 只读深度分析: {query}")
        
        # 提取约束
        constraints = self.agent_rl.query_rewriter.extract_readonly_constraints(query)
        analysis_params = self.agent_rl.query_rewriter.extract_analysis_params(query)
        
        # 验证只读模式
        if not constraints["readonly"]:
            print("  [警告] 未检测到只读标记，但仍以只读模式执行")
            constraints["readonly"] = True
        
        # 执行分析（完整流水线，只读）
        result = self._execute_file_analysis(state, query, constraints, analysis_params)
        
        state.add_evidence(RetrievalResult(
            documents=[{"content": result, "source": "file_analysis", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()
        
        reward_key = "query_success" if result else "insufficient_evidence"
        done = True
        reason = "文件分析完成" if result else "文件分析失败"
        
        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        
        state.terminate(result)

    def _execute_file_analysis(self, state: AgentState, query: str, 
                               constraints: Dict, params: Dict) -> str:
        """执行文件分析（只读，走专业流水线：解析/AST/依赖/决策）。"""
        from review_pipeline import ReviewPipeline
        from content_parser import parse_file
        from relation_analyzer import analyze_relations
        from decision_engine import decide
        from git_history import collect_git_history, format_git_summary
        from ignore_rules import create_default_engine
        
        lines = ["  [文件分析报告]", "  模式: 只读（禁止修改/移动/删除/重命名/覆盖）", ""]
        
        ignore = create_default_engine(self.corpus.root)
        ignore.load_all()
        for d in constraints.get("exclude_dirs", []):
            ignore.add_rule(str(d).strip("/") + "/", source="custom")
        
        pipe = ReviewPipeline(self.corpus.root, max_rounds=state.max_rounds)
        # 分析阶段：collect + git + parse + analyze + decide（不写报告）
        pipe.run(
            query=query,
            stages=["ignore", "collect", "git", "parse", "analyze", "decide"],
            similarity_threshold=float(params.get("similarity_threshold", 0.82)),
            complexity_threshold=int(params.get("complexity_threshold", 15)),
            git_days=45,
        )
        st = pipe.state
        coll = st.memory.get("collection")
        parsed = st.memory.get("parsed", [])
        relations = st.memory.get("relations")
        decision = st.memory.get("decision")
        git = st.memory.get("git")
        
        if coll is None or decision is None:
            return "  [文件分析失败] 采集/决策阶段未完成: " + (st.terminate_reason or "; ".join(st.errors[:3]))
        
        from collections import Counter
        lines.append(f"  扫描文件数: {len(coll.files)}（可读 {sum(1 for f in coll.files if f.readable)}）")
        lines.append(f"  跳过: {len(coll.skipped)}  子项目: {', '.join(coll.subprojects) or '无'}")
        lines.append(f"  忽略源: {', '.join(coll.ignore_sources) or '无'}")
        lines.append(f"  压缩包: {len(coll.archives)}  包内索引: {len(coll.archive_index)} 条")
        lines.append("")
        
        ext_counter = Counter((f.language_ext or "无扩展名") for f in coll.files)
        lines.append("  文件类型分布:")
        for ext, count in ext_counter.most_common(12):
            lines.append(f"    {ext}: {count} 个")
        lines.append("")
        
        # 语言分组 / 目录分组 / 风险分组
        lines.append("  按语言分组（发现数）:")
        for lang, ids in sorted(decision.groups_by_language.items(), key=lambda x: -len(x[1])):
            lines.append(f"    {lang}: {len(ids)}")
        lines.append("  按风险分组:")
        for lv in ("HIGH", "MEDIUM", "LOW"):
            lines.append(f"    {lv}: {len(decision.groups_by_risk.get(lv, []))}")
        lines.append("")
        
        # 条件交集
        lines.append("  条件交集:")
        if decision.condition_intersections:
            for name, paths in decision.condition_intersections.items():
                lines.append(f"    {name}: {', '.join(paths[:8])}")
        else:
            lines.append("    无")
        lines.append("")
        
        # 活跃 TODO（前 20）
        todos = relations.todo_priority if relations else []
        lines.append(f"  活跃 TODO/FIXME（共 {len(todos)}，展示前 20）:")
        for t in todos[:20]:
            lines.append(f"    {t['file']}:{t['line']} [{t['marker']}] {t['text'][:100]}")
        if not todos:
            lines.append("    无")
        lines.append("")
        
        # 高复杂度
        complex_fns = relations.complex_functions if relations else []
        lines.append(f"  复杂度≥15 函数（共 {len(complex_fns)}）:")
        for h in complex_fns[:15]:
            lines.append(f"    {h.file}:{h.line} {h.name} = {h.complexity}")
        if not complex_fns:
            lines.append("    无")
        lines.append("")
        
        # 依赖图统计
        if relations:
            lines.append(f"  依赖图: {len(relations.dependency_graph)} 节点有出边, "
                         f"循环 {len(relations.circular_deps)}, 孤儿 {len(relations.orphan_modules)}")
            if relations.circular_deps:
                for c in relations.circular_deps[:5]:
                    lines.append("    循环: " + " -> ".join(c))
            lines.append(f"  相似度≥0.82 重复对: {len(relations.duplicate_pairs)}")
            lines.append(f"  交叉引用: {len(relations.cross_references)}")
            if relations.cross_references:
                for x in relations.cross_references[:10]:
                    lines.append(f"    [{x.kind}] {x.left} ↔ {x.right}")
        lines.append("")
        
        # Git
        lines.append("  Git 历史（最近45天）:")
        for ln in format_git_summary(git).splitlines():
            lines.append(f"    {ln}")
        lines.append("")
        
        # 跳过记录样例
        lines.append(f"  跳过记录（{len(coll.skipped)} 条，按类别）:")
        cat_counter = Counter(s.category for s in coll.skipped)
        for cat, n in cat_counter.most_common():
            lines.append(f"    {cat}: {n}")
        for s in coll.skipped[:10]:
            lines.append(f"    - [{s.category}] {s.path}: {s.reason}")
        lines.append("")
        
        # 分析参数回显
        if params.get("complexity_threshold"):
            lines.append(f"  复杂度阈值: {params['complexity_threshold']}")
        if params.get("similarity_threshold"):
            lines.append(f"  相似度阈值: {params['similarity_threshold']}")
        if params.get("include_git_log"):
            lines.append("  已包含 git log 过滤")
        
        # 运行时状态
        lines.append("")
        lines.append(f"  运行时: 回合 {st.rounds}/{st.max_rounds}, "
                     f"工具调用 {len(st.tool_calls)}, 阶段 {', '.join(st.stages_done)}")
        return "\n".join(lines)

    def _route_file_scan(self, state: AgentState, query: str):
        """路由到文件扫描分支。"""
        print(f"\n>>> [文件扫描] 只读扫描: {query}")
        
        # 提取约束
        constraints = self.agent_rl.query_rewriter.extract_readonly_constraints(query)
        
        # 执行扫描
        result = self._execute_file_scan(state, query, constraints)
        
        state.add_evidence(RetrievalResult(
            documents=[{"content": result, "source": "file_scan", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()
        
        reward_key = "query_success" if result else "insufficient_evidence"
        done = True
        reason = "文件扫描完成" if result else "文件扫描失败"
        
        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        
        state.terminate(result)

    def _execute_file_scan(self, state: AgentState, query: str, constraints: Dict) -> str:
        """执行文件扫描（只读，输出每个文件路径 + 跳过原因）。"""
        from review_pipeline import ReviewPipeline
        from ignore_rules import create_default_engine
        
        lines = ["  [文件扫描报告]", "  模式: 只读"]
        
        ignore = create_default_engine(self.corpus.root)
        ignore.load_all()
        for d in constraints.get("exclude_dirs", []):
            ignore.add_rule(str(d).strip("/") + "/", source="custom")
        for f in constraints.get("ignore_files", []):
            ignore.add_rule(str(f), source="custom")
        
        pipe = ReviewPipeline(self.corpus.root, max_rounds=state.max_rounds)
        pipe.run(
            query=query,
            stages=["collect"],  # 复用 ignore 内部加载会覆盖？—— 显式传 ignore
        )
        # collect 阶段内部会自建 ignore；为应用用户约束，直接调用工具
        from review_collector import collect as _collect
        coll = _collect(self.corpus.root, ignore=ignore, include_archives=True, read_contents=True)
        pipe.state.scanned_files = [f.path for f in coll.files]
        pipe.state.skipped = [
            {'path': s.path, 'reason': s.reason, 'category': s.category}
            for s in coll.skipped
        ]
        pipe.state.stages_done.append('collect')
        pipe.state.terminated = True
        pipe.state.terminate_reason = '扫描完成'
        
        lines.append(f"  项目根: {coll.root}")
        lines.append(f"  子项目: {', '.join(coll.subprojects) or '无'}")
        lines.append(f"  忽略源: {', '.join(coll.ignore_sources) or '无'}")
        lines.append(f"  扫描文件数: {len(coll.files)}")
        lines.append(f"  可读: {sum(1 for f in coll.files if f.readable)}  "
                     f"二进制/其他: {sum(1 for f in coll.files if not f.readable)}")
        lines.append(f"  跳过: {len(coll.skipped)}  压缩包: {len(coll.archives)}")
        lines.append("")
        lines.append("  文件列表（每个文件路径）:")
        for f in sorted(coll.files, key=lambda x: x.path):
            mark = "" if f.readable else " [非文本]"
            arc = " (包内)" if f.in_archive else ""
            lines.append(f"    {f.path}  {f.size}B{mark}{arc}")
        lines.append("")
        lines.append("  跳过记录（含原因）:")
        if coll.skipped:
            from collections import Counter
            cats = Counter(s.category for s in coll.skipped)
            lines.append(f"    汇总: {dict(cats)}")
            for s in coll.skipped[:40]:
                lines.append(f"    - [{s.category}] {s.path}: {s.reason}")
            if len(coll.skipped) > 40:
                lines.append(f"    ... 另有 {len(coll.skipped) - 40} 条")
        else:
            lines.append("    无")
        lines.append("")
        lines.append(f"  运行时: 回合 {pipe.state.rounds}/{pipe.state.max_rounds}, "
                     f"已扫描 {len(pipe.state.scanned_files)}, 已跳过 {len(pipe.state.skipped)}")
        return "\n".join(lines)

    def _route_report_generate(self, state: AgentState, query: str):
        """路由到报告生成分支（完整流水线 + 唯一目录 + MD/JSON/CSV + 校验）。"""
        print(f"\n>>> [报告生成] 生成审查报告: {query}")
        
        # 提取约束
        constraints = self.agent_rl.query_rewriter.extract_readonly_constraints(query)
        analysis_params = self.agent_rl.query_rewriter.extract_analysis_params(query)
        
        # 检测输出格式（流水线始终产出 MD/JSON/CSV 全套）
        output_format = "markdown"
        q = query.lower()
        if "json" in q:
            output_format = "json"
        elif "csv" in q:
            output_format = "csv"
        
        # 生成报告（写入唯一 review_* 目录）
        result = self._execute_report_generate(state, query, constraints, analysis_params, output_format)
        
        state.add_evidence(RetrievalResult(
            documents=[{"content": result, "source": "report_generate", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()
        
        reward_key = "query_success" if result and "失败" not in result[:80] else "insufficient_evidence"
        done = True
        reason = "报告生成完成" if reward_key == "query_success" else "报告生成失败"
        
        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        
        state.terminate(result)

    def _route_filename_query(self, state: AgentState, query: str):
        """路由到文件名搜索分支（按名称/时间筛选文件）。"""
        print(f"\n>>> [文件名搜索] 搜索文件名: {query}")
        
        # 提取时间范围槽位
        time_range = state.slots.get("time_range") or self._extract_time_range(query)
        file_ext = self._extract_file_extension(query)
        
        result = self._execute_filename_query(state, query, time_range, file_ext)
        
        state.add_evidence(RetrievalResult(
            documents=[{"content": result, "source": "filename_query", "score": 1.0}],
            score=1.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()
        
        reward_key = "query_success" if result else "insufficient_evidence"
        done = True
        reason = "文件名搜索完成" if result else "文件名搜索未找到匹配文件"
        
        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")
        
        state.terminate(result)

    def _route_filelist_search(self, state: AgentState, query: str):
        """路由到文件列表搜索分支（指定根路径 + 云 LLM ReAct 扫描，只读）。

        缺根路径不直接执行：走澄清闭环，用户补路径后重新进入意图识别。
        """
        spec = self.filelist_handler.parse_spec(query, state.slots)
        if not spec.root:
            print("\n>>> [文件列表搜索] 未识别到根路径，触发澄清")
            self._route_clarification(
                state, self.filelist_handler.clarification_for(query))
            return

        print(f"\n>>> [文件列表搜索] 只读扫描: {query}")
        result = self.filelist_handler.execute(state, spec)
        found = self.filelist_handler.found

        state.add_evidence(RetrievalResult(
            documents=[{
                "content": result,
                "source": "filelist_search",
                "score": 1.0 if found else 0.0,
            }],
            score=1.0 if found else 0.0,
            query_used=query,
            round_num=state.current_round,
        ))
        state.increment_round()

        reward_key = "query_success" if found else "insufficient_evidence"
        done = True
        reason = "文件列表检索完成" if found else "文件列表检索未找到匹配文件"

        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(result)

    def _extract_time_range(self, query: str) -> str:
        """从查询中提取时间范围。"""
        import re
        q = query.lower()
        if "今天" in q:
            return "today"
        if "昨天" in q:
            return "yesterday"
        if "本周" in q or "这周" in q:
            return "this_week"
        if "本月" in q or "这个月" in q:
            return "this_month"
        if "最近一天" in q or "昨天" in q:
            return "recent_1day"
        if "最近一周" in q or "最近七天" in q:
            return "recent_1week"
        if "最近一个月" in q or "最近30天" in q:
            return "recent_1month"
        m = re.search(r"最近(\d+)(分钟|小时|天|周|月)", q)
        if m:
            return f"recent_{m.group(1)}{m.group(2)}"
        return "recent_1day"  # 默认最近一天

    def _extract_file_extension(self, query: str) -> Optional[str]:
        """从查询中提取文件扩展名筛选。"""
        import re
        m = re.search(r"\.([a-zA-Z0-9]+)\s*(?:文件|的)?", query)
        if m:
            return f".{m.group(1).lower()}"
        return None

    def _execute_filename_query(self, state: AgentState, query: str,
                                 time_range: str, file_ext: Optional[str]) -> str:
        """执行文件名搜索（按时间筛选）。"""
        import os
        import time as _time
        from datetime import datetime, timedelta
        
        results = []
        results.append("  [文件名搜索结果]\n")
        
        # 计算时间阈值
        now = _time.time()
        if time_range == "today":
            start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        elif time_range == "yesterday":
            start = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0).timestamp()
            end = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        elif time_range == "this_week":
            start = (datetime.now() - timedelta(days=datetime.now().weekday())).replace(
                hour=0, minute=0, second=0).timestamp()
        elif time_range == "this_month":
            start = datetime.now().replace(day=1, hour=0, minute=0, second=0).timestamp()
        else:
            start = now - 86400  # 默认最近一天
        
        matched_files = []
        search_root = self.corpus.root
        
        for root, dirs, files in os.walk(search_root):
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(file_path)
                    if mtime >= start:
                        # 扩展名过滤
                        if file_ext and not f.lower().endswith(file_ext):
                            continue
                        rel_path = os.path.relpath(file_path, search_root)
                        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                        matched_files.append((rel_path, mtime_str))
                except (OSError, PermissionError):
                    continue
        
        # 按修改时间降序排列
        matched_files.sort(key=lambda x: x[1], reverse=True)
        
        results.append(f"  搜索范围: {search_root}\n")
        results.append(f"  时间筛选: {time_range}\n")
        if file_ext:
            results.append(f"  文件类型: {file_ext}\n")
        results.append(f"  找到 {len(matched_files)} 个匹配文件\n\n")
        
        for path, mtime in matched_files[:30]:
            results.append(f"    {path}  ({mtime})\n")
        
        if len(matched_files) > 30:
            results.append(f"    ... (共 {len(matched_files)} 个文件，仅显示前30个)\n")
        
        if not matched_files:
            results.append("  未找到匹配文件。\n")
            results.append("  建议: 尝试扩大时间范围或去掉文件类型限制。\n")
        
        return "".join(results)

    def _execute_report_generate(self, state: AgentState, query: str, 
                                 constraints: Dict, params: Dict, output_format: str) -> str:
        """生成审查报告：调用 ReviewPipeline 写出唯一目录全套产物。"""
        from review_pipeline import ReviewPipeline
        from ignore_rules import create_default_engine
        
        ignore = create_default_engine(self.corpus.root)
        ignore.load_all()
        for d in constraints.get("exclude_dirs", []):
            ignore.add_rule(str(d).strip("/") + "/", source="custom")
        
        pipe = ReviewPipeline(self.corpus.root, max_rounds=state.max_rounds)
        snapshot = pipe.run(
            query=query,
            stages=["ignore", "collect", "git", "parse", "analyze", "decide", "report", "verify"],
            similarity_threshold=float(params.get("similarity_threshold", 0.82)),
            complexity_threshold=int(params.get("complexity_threshold", 15)),
            git_days=45 if params.get("include_git_log") or True else 45,
        )
        
        report_dir = snapshot.get("report_dir") or ""
        decision = pipe.state.memory.get("decision")
        coll = pipe.state.memory.get("collection")
        validation = pipe.state.memory.get("validation") or {}
        
        if not report_dir or decision is None:
            return "  [报告生成失败] " + (snapshot.get("terminate_reason") or "; ".join(
                (pipe.state.errors or ["未知错误"])[:3]))
        
        s = decision.summary
        lines = ["  [审查报告已生成]", ""]
        lines.append(f"  目录: {report_dir}")
        lines.append(f"  校验: {'通过' if snapshot.get('validated') else '未通过'}")
        if not snapshot.get("validated"):
            for c in validation.get("checks", []):
                if not c.get("ok"):
                    lines.append(f"    ✗ {c.get('name')}: {c.get('detail')}")
        lines.append("")
        lines.append(f"  发现: HIGH {s.get('high', 0)} / MEDIUM {s.get('medium', 0)} / LOW {s.get('low', 0)}")
        lines.append(f"  文件: {s.get('files_analyzed', 0)} 可解析, "
                     f"跳过 {len(coll.skipped) if coll else 0}, "
                     f"压缩包 {len(coll.archives) if coll else 0}")
        lines.append(f"  条件交集: {len(decision.condition_intersections)} 组")
        lines.append(f"  运行时: 回合 {snapshot.get('rounds')}, "
                     f"工具 {len(snapshot.get('tools', []))}, "
                     f"阶段 {', '.join(snapshot.get('stages_done', []))}")
        lines.append("")
        lines.append("  产物:")
        for name in ("report.md", "index.json", "findings.csv", "files.csv",
                     "archive_index.csv", "skips.csv", "dependency_graph.json"):
            lines.append(f"    - {name}")
        lines.append(f"    - summaries/ ({len((pipe.state.memory.get('artifacts') or {}).get('summaries', []))} 个只读摘要)")
        lines.append("")
        lines.append("  优先修复（Top 10）:")
        for f in decision.findings[:10]:
            loc = f"{f.file}:{f.line}" if f.line else f.file
            lines.append(f"    {f.priority}. [{f.severity}/{f.category}] {loc} — {f.description[:90]}")
        
        # 按 output_format 附加预览
        if output_format == "json":
            import json as _json
            preview = {
                "report_dir": report_dir,
                "summary": s,
                "condition_intersections": decision.condition_intersections,
                "top_findings": [
                    {"id": f.id, "severity": f.severity, "file": f.file,
                     "line": f.line, "description": f.description}
                    for f in decision.findings[:20]
                ],
                "validated": snapshot.get("validated"),
            }
            lines.append("")
            lines.append("  JSON 预览:")
            lines.append(_json.dumps(preview, ensure_ascii=False, indent=2))
        elif output_format == "csv":
            lines.append("")
            lines.append("  CSV 位于 findings.csv / files.csv / skips.csv / archive_index.csv")
        
        return "\n".join(lines)

    def _create_report_directory(self, state: AgentState) -> str:
        """创建报告目录（唯一 review_YYYYMMDD_HHmmss_<hash>，不覆盖）。"""
        from report_writer import make_review_dir
        report_dir = make_review_dir(self.corpus.root, fingerprint_src=str(state.current_round))
        print(f"  [报告目录] 已创建: {report_dir}")
        return report_dir

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

        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(ocr_result)

    def _route_query_metadata(self, state: AgentState, query: str):
        """路由到文件元数据查询。"""
        print(f"\n>>> [文件信息] 查询文件元数据: {query}")

        target = self._slot_file_target(state, query)
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

        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(content)

    def _route_query_content(self, state: AgentState, query: str):
        """路由到文件内容查询。"""
        print(f"\n>>> [文件内容] 读取文件内容: {query}")

        target = self._slot_file_target(state, query)
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

        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(display_content)

    def _route_query_summary(self, state: AgentState, query: str):
        """路由到内容总结。"""
        print(f"\n>>> [内容总结] 总结文件内容: {query}")

        target = self._slot_file_target(state, query)
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

        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(summary)

    def _route_query_translate(self, state: AgentState, query: str):
        """路由到内容翻译。"""
        print(f"\n>>> [内容翻译] 翻译文件内容: {query}")

        target = self._slot_file_target(state, query)
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

        reward = self.agent_rl.compute_reward(reward_key, state)
        self.agent_rl.learn(reward, state, done)
        print(f"  [RL奖励] {reward:+.1f} ({reward_key})")
        print(f"  [终止判定] {reason}")

        state.terminate(translated)

    def _route_query_extract(self, state: AgentState, query: str):
        """路由到字段提取。"""
        print(f"\n>>> [字段提取] 提取文件信息: {query}")

        target = self._slot_file_target(state, query)
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

        reward = self.agent_rl.compute_reward(reward_key, state)
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