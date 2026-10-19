import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from agent_state import AgentState, RetrievalResult
from query_rewriter import QueryRewriter
from file_corpus import FileCorpus
from evidence_ledger import EvidenceLedger
from rl_config import get_llm_context_budget


# 合法动作集合：解析器据此判定 LLM 输出是否越界并降级
VALID_ACTIONS = {"search", "refine", "verify", "bridge", "answer"}


REACT_SYSTEM_PROMPT = """\
你是 AgentRL 深度检索 ReAct 引擎，通过"思考-行动-观察"循环逐步补齐证据。

## 当前状态
- 已完成检索轮次: {round_num}/{max_rounds}
- 已收集文档数: {doc_count}

## 可选动作（每次只输出一个 JSON 对象）
1) search —— 开启一个全新角度的检索
   {{"thought": "...", "action": "search", "search_query": "..."}}
2) refine —— 对上一角度换措辞或同义改写后重搜
   {{"thought": "...", "action": "refine", "search_query": "..."}}
3) bridge —— 针对用户消息中"未覆盖方面"补缺口
   {{"thought": "...", "action": "bridge", "target_aspect": "性能", "search_query": "..."}}
4) verify —— 对"主张列表"中单来源的重要主张做交叉验证
   {{"thought": "...", "action": "verify", "target_claim": 1, "search_query": "..."}}
   （target_claim 是用户消息里编号主张列表的 1-based 序号）
5) answer —— 证据充分时给出最终回答
   {{"thought": "...", "action": "answer", "answer": "..."}}

## 决策规则
- 存在"未覆盖方面"时优先 bridge，把缺口补上再考虑收口。
- 仅当某条重要主张只有单一来源时才用 verify 交叉验证，不要滥用。
- 覆盖完整且置信度足够高时用 answer 收口。
- 到达最后一轮时，必须用 answer 基于现有证据作答，不得再发起检索。
"""


class ReActLLMClient:
    """用于 ReAct 循环的 LLM 客户端。

    端点链复用 llm_intent.LLMIntentClassifier：读取 config/app.yaml 的 intent_llm 段，
    主端点（provider/model）未就绪时自动降级到 intent_llm.fallback 备端点
    （如本地 Ollama），因此"主端点没配 key"不再等于"LLM 永远不可用"。
    """

    def __init__(self):
        self._chain = None
        self._init_error = ""

    def _ensure(self):
        if self._chain is None and not self._init_error:
            try:
                from llm_intent import LLMIntentClassifier
                self._chain = LLMIntentClassifier(quiet=True)
            except Exception as e:
                self._init_error = f"端点链初始化失败: {e}"
        return self._chain

    @property
    def available(self) -> bool:
        chain = self._ensure()
        return bool(chain and chain.available)

    @property
    def status(self) -> str:
        chain = self._ensure()
        if chain:
            return chain.status
        return "error" if self._init_error else "unconfigured"

    @property
    def endpoint(self) -> str:
        chain = self._ensure()
        if not chain:
            return self._init_error or "-"
        return f"{chain.provider}/{chain.model}"

    @property
    def reason(self) -> str:
        """端点链不可用时的可诊断原因（空串表示可用）。"""
        if self._init_error:
            return self._init_error
        chain = self._ensure()
        if not chain:
            return "端点链未加载"
        if chain.available:
            return ""
        return chain.status_reason or "未配置可用端点（主端点与 fallback 均未就绪）"

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        chain = self._ensure()
        if not chain or not chain.available:
            return None
        return chain.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=600,
            temperature=0.2,
        )


def _extract_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


class DeepSearchHandler:
    """
    深度检索分支（ReAct 模式）：LLM 驱动的思考-行动-观察循环。

    每一轮：
      1. Thought  —— LLM 分析当前证据，判断是否充分，决定下一步
      2. Action   —— 如果不足：search（指定搜索关键词）；如果充分：answer（生成回答）
      3. Observation —— FAISS 检索结果（或最终回答）
    直到 LLM 判断证据充分 或 达到最大轮次。
    """

    def __init__(self, corpus: FileCorpus = None, max_rounds: int = 5, top_k: int = 5):
        self.corpus = corpus or FileCorpus()
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.query_rewriter = QueryRewriter()
        self.llm = ReActLLMClient()
        self.react_steps: List[Dict[str, str]] = []
        # 本次任务已发出的检索查询（归一化），用于杜绝"同一查询重复检索"
        self._searched_queries: Set[str] = set()
        self.intent_slots: Dict[str, Any] = {}
        # 证据台账：方面覆盖 + 主张置信度，跨轮累积
        self.ledger = EvidenceLedger()

    def _build_user_prompt(self, query: str, state: AgentState) -> str:
        # 单一有界上下文块：证据随 build_llm_context 在 token 预算内裁剪，
        # 保留 [证据i] 标记（answer_generator 降级路径依赖该标记）。
        structured = state.build_llm_context(
            budget_tokens=get_llm_context_budget(), include_evidence=True
        )
        searched = sorted(self._searched_queries) or ["（尚无）"]
        claim_lines = []
        for rank, (_, claim) in enumerate(self.ledger.top_claims(5), 1):
            sources = "|".join(claim.sources) if claim.sources else "无"
            claim_lines.append(
                f"{rank}. (置信度 {claim.confidence:.2f}, 来源 {sources}) {claim.text}"
            )
        claims_block = "\n".join(claim_lines) if claim_lines else "（暂无主张）"
        return (
            f"用户原始问题: {query}\n"
            f"已解析意图: {self.intent_slots.get('task_type', '')}；"
            f"核心主题: {self.intent_slots.get('core_topics', [])}\n\n"
            f"=== 结构化上下文（槽位/证据/对话历史/操作历史） ===\n{structured}\n\n"
            f"=== 已检索过的查询（禁止重复） ===\n"
            + "\n".join(f"- {q}" for q in searched) + "\n\n"
            f"=== 证据台账 ===\n{self.ledger.summary()}\n"
            f"编号主张列表（verify 的 target_claim 即此 1-based 序号）：\n{claims_block}\n\n"
            f"当前是第 {state.current_round}/{self.max_rounds} 轮。\n"
            f"请分析证据缺口，输出下一步行动的 JSON。"
        )

    def _parse_react_action(self, llm_output: str) -> Tuple[str, str, str, Dict[str, Any]]:
        """解析 LLM 输出，返回 (action, thought, payload, meta)。

        meta 携带动作语义：search/refine/bridge 的 aspect、verify 的 claim_index
        （JSON 的 1-based target_claim 转 0-based 内部下标）。
        """
        data = _extract_json(llm_output)
        if data and isinstance(data, dict):
            action = str(data.get("action", "answer")).lower()
            thought = data.get("thought", "")
            if action not in VALID_ACTIONS:
                # 越界动作降级：有非空 answer 视为作答，否则退回检索
                action = "answer" if data.get("answer") else "search"
            if action in ("search", "refine", "bridge"):
                return action, thought, data.get("search_query", ""), {
                    "aspect": data.get("target_aspect", "")
                }
            if action == "verify":
                idx = None
                try:
                    raw = int(data.get("target_claim"))
                    if raw >= 1:
                        idx = raw - 1
                except (TypeError, ValueError):
                    idx = None
                return action, thought, data.get("search_query", ""), {
                    "aspect": "", "claim_index": idx
                }
            return "answer", thought, data.get("answer", ""), {"aspect": ""}
        if "answer" in llm_output.lower():
            return "answer", llm_output[:200], llm_output, {"aspect": ""}
        return "search", "尝试基于关键词搜索", "", {"aspect": ""}

    # 中文停用字：只承担语法功能，其二元组不能当作"证据命中"探针
    _STOP_CHARS = set(
        "的了是在和与或及对把被就都很也有个们这那么但而且你您我他她它"
        "请帮我一下吗呢吧啊呀么嘛之其于以从向给为所并将再又还只则即可"
        "已未不无非没别"
    )

    # 查询里出现的分析维度 → 子查询角度（把原始问题拆成多轮可检索的方向）
    _ASPECT_HINTS = (
        ("架构", ("架构", "结构", "模块", "组织", "目录", "分层", "框架")),
        ("性能", ("性能", "瓶颈", "效率", "耗时", "延迟", "吞吐", "优化")),
        ("依赖", ("依赖", "import", "require", "引用", "耦合", "调用")),
        ("文档", ("文档", "注释", "readme", "docstring")),
        ("配置", ("配置", "config", "yaml", "环境变量", "env")),
        ("质量", ("质量", "复杂度", "重复", "坏味道", "规范", "风格")),
        ("测试", ("测试", "test", "覆盖率")),
        ("安全", ("安全", "密钥", "secret", "漏洞", "风险")),
        ("历史", ("git", "提交", "历史", "修改记录")),
    )

    _LEAD_VERB_RE = re.compile(
        r"^(?:请|帮我|帮忙|能不能|可以|麻烦|需要)?"
        r"(?:一下)?"
        r"(?:深度|深入|综合|全面|详细|仔细|系统)?"
        r"(?:分析|研究|评估|审查|检查|总结|汇总|对比|比较|查找|搜索|查询|查看|探索|讨论)"
    )

    @staticmethod
    def _norm_query(text: str) -> str:
        """查询归一化：判定"这个查询是否已经检索过"。"""
        return re.sub(r"[\s，。！？、,.!?:;；:（）()【】\[\]「」\"'`]+", "", (text or "")).lower()

    @classmethod
    def _probes(cls, text: str) -> set:
        """词面探针：ASCII 词 + 中文二元组（无需分词器即可做覆盖率判定）。"""
        text = (text or "").lower()
        probes = set(re.findall(r"[a-z0-9_.\-]{3,}", text))
        han = re.findall(r"[\u4e00-\u9fa5]", text)
        for i in range(len(han) - 1):
            bg = han[i] + han[i + 1]
            if all(c in cls._STOP_CHARS for c in bg):
                continue
            probes.add(bg)
        return probes

    @staticmethod
    def _detect_aspects(query: str) -> List[str]:
        """查询中显式提到的分析维度（性能/架构/依赖...）。"""
        q = (query or "").lower()
        return [name for name, hints in DeepSearchHandler._ASPECT_HINTS
                if any(h in q for h in hints)]

    def _topic_of(self, query: str) -> str:
        """去掉动作动词与语气词，只留主题名词短语，用于拼接子查询。"""
        core = self._extract_core_query(query)
        topic = self._LEAD_VERB_RE.sub("", (core or "").strip())
        topic = re.sub(r"(可以吗|好吗|一下|？|\?|。)+$", "", topic).strip()
        return topic if topic else (core or "该项目")

    def _build_search_plan(self, query: str) -> List[str]:
        """把原始问题拆成有序检索角度 —— 与原问题强相关，而非固定模板清单。"""
        topic = self._topic_of(query)
        try:
            targets = self.query_rewriter.extract_targets(query) or []
        except Exception:
            targets = []
        try:
            _, entities, _ = self.query_rewriter.rewrite(query)
        except Exception:
            entities = []

        plan: List[str] = [topic]
        if len(targets) >= 2:
            plan.append(f"对比 {targets[0]} 和 {targets[1]} 的差异")
        for aspect in self._detect_aspects(query):
            plan.append(f"{topic} 的{aspect}")
        for ent in entities:
            if len(plan) >= self.max_rounds:
                break
            cand = f"{ent} 的内容与结构"
            if cand not in plan:
                plan.append(cand)
        for filler in ("相关的文件清单", "配置与依赖关系", "实现与关键内容", "潜在风险与缺口"):
            if len(plan) >= max(self.max_rounds, 3):
                break
            plan.append(f"{topic} {filler}")

        seen, ordered = set(), []
        for sq in plan:
            key = self._norm_query(sq)
            if key and key not in seen:
                seen.add(key)
                ordered.append(sq)
        return ordered[:max(self.max_rounds, 1)]

    def _plan_gaps(self, query: str, state: AgentState) -> List[Tuple[str, bool]]:
        """各检索角度的覆盖情况：探针在已收集证据（路径+正文）中的命中率 ≥ 50% 视为已覆盖。"""
        plan = self._build_search_plan(query)
        evidence_text = " ".join(
            f"{d.get('path', '')} {d.get('content', '')}"
            for ev in state.evidence for d in ev.documents
        ).lower()

        gaps: List[Tuple[str, bool]] = []
        for sq in plan:
            probes = self._probes(sq)
            if not probes or not evidence_text:
                gaps.append((sq, False))
                continue
            hits = sum(1 for p in probes if p in evidence_text)
            gaps.append((sq, hits / len(probes) >= 0.5))
        return gaps

    def _select_gap_query(self, query: str, state: AgentState
                          ) -> Tuple[Optional[str], List[Tuple[str, bool]]]:
        """挑下一个检索角度：只取「既没被证据覆盖、也没检索过」的 —— 循环不退化的关键。"""
        gaps = self._plan_gaps(query, state)
        for sq, covered in gaps:
            if covered or self._norm_query(sq) in self._searched_queries:
                continue
            return sq, gaps
        return None, gaps

    def _rule_based_fallback(self, query: str, state: AgentState
                             ) -> Tuple[str, str, str, Dict[str, Any]]:
        """LLM 不可用时的确定性 ReAct 决策：证据缺口驱动 + 查询去重 + 抽取式综合。

        返回 (action, thought, payload, meta)；thought 是真实的缺口分析，而不是把待执行
        的动作倒填成"思考"。缺口查询命中已知方面名时升级为 bridge，否则为 search；
        无缺口但存在单来源低置信主张时升级为 verify。
        """
        next_query, gaps = self._select_gap_query(query, state)
        total = len(gaps)
        covered_n = sum(1 for _, cov in gaps if cov)
        tried = len(self._searched_queries)
        evidence_n = len(state.evidence)

        if next_query:
            missing = [sq for sq, cov in gaps if not cov][:3]
            thought = (
                f"缺口分析：{total} 个检索角度已覆盖 {covered_n} 个，已检索 {tried} 次；"
                f"尚未覆盖 {missing} → 本轮针对「{next_query}」补充证据"
            )
            aspects = self._detect_aspects(next_query)
            if aspects:
                return "bridge", thought, next_query, {"aspect": aspects[0]}
            return "search", thought, next_query, {"aspect": ""}

        # 无新角度可检索：若存在单来源且低置信的重要主张，先交叉验证再收口
        if self.ledger.claims and self.ledger.min_confidence() < 0.55:
            for idx, claim in enumerate(self.ledger.claims):
                if len(claim.sources) == 1 and \
                        self._norm_query(claim.text) not in self._searched_queries:
                    thought = (
                        f"缺口分析：检索角度已穷尽，但主张「{claim.text[:40]}」"
                        f"仅单一来源且整体置信度偏低 → 交叉验证该主张"
                    )
                    return "verify", thought, claim.text, {
                        "aspect": claim.aspect, "claim_index": idx
                    }

        if covered_n:
            reason = f"全部检索角度已被证据覆盖（{covered_n}/{total}）"
        else:
            reason = f"已穷尽 {total} 个检索角度且证据未命中（{tried} 次检索无新增方向）"
        thought = f"缺口分析：{reason}，基于现有 {evidence_n} 轮证据综合回答"
        return "answer", thought, self._synthesize_answer(query, gaps, state, self.ledger), {
            "aspect": ""
        }

    def _synthesize_answer(self, query: str, gaps: List[Tuple[str, bool]],
                           state: AgentState,
                           ledger: Optional[EvidenceLedger] = None) -> str:
        """综合回答：台账有主张时按主张聚合（含置信度与未覆盖告警），否则退回抽取式。"""
        if ledger is not None and ledger.claims:
            top = ledger.top_claims(5)
            doc_paths: List[str] = []
            for _, claim in top:
                for src in claim.sources:
                    if src and src not in doc_paths:
                        doc_paths.append(src)
            lines = [
                f"基于 {len(state.evidence)} 轮检索、{len(doc_paths)} 个不同文件、"
                f"{len(ledger.claims)} 条主张综合如下"
                f"（整体置信度 {ledger.min_confidence(3):.2f}）：",
                "",
            ]
            uncovered = ledger.required_uncovered
            if uncovered:
                lines.append(f"⚠️ 证据不完整：以下方面未覆盖 — {'、'.join(uncovered)}")
                lines.append("")
            lines.append("**按主张聚合**")
            for rank, (_, claim) in enumerate(top, 1):
                sources = "|".join(claim.sources) if claim.sources else "无"
                lines.append(
                    f"{rank}. (得分 {claim.score:.4f}, 置信度 {claim.confidence:.2f}, "
                    f"来源 {sources}) {claim.text}"
                )
            lines += ["", "**检索角度覆盖**"]
            for sq, cov in gaps:
                lines.append(f"- [{'已覆盖' if cov else '未覆盖'}] {sq}")
            lines += ["", "**引用来源**"]
            lines += [f"- {p}" for p in doc_paths] or ["- （无）"]
            return "\n".join(lines)

        # 抽取式综合：同文件去重取最高分 → 每个文件挑最相关的句子 → 结构化输出。
        best: Dict[str, Dict] = {}
        for ev in state.evidence:
            for d in ev.documents:
                path = d.get("path") or d.get("source") or ""
                if not path:
                    continue
                cur = best.get(path)
                if cur is None or d.get("score", 0.0) > cur.get("score", 0.0):
                    item = dict(d)
                    item["_round"] = ev.round_num
                    best[path] = item
        docs = sorted(best.values(), key=lambda x: x.get("score", 0.0), reverse=True)

        if not docs:
            tried = "、".join(sq for sq, _ in gaps) or "无"
            return (f"经过 {len(state.evidence)} 轮检索，语料库中未找到与「{query}」"
                    f"直接相关的证据。（已尝试角度：{tried}）")

        probes = self._probes(query)
        lines = [
            f"基于 {len(state.evidence)} 轮检索、{len(docs)} 个不同文件的证据综合如下：",
            "",
            "**关键证据**",
        ]
        for i, d in enumerate(docs[:5], 1):
            sentence = self._best_sentence(d.get("content", ""), probes)
            lines.append(f"{i}. `{d.get('path', '?')}`"
                         f"（得分 {d.get('score', 0.0):.4f}，第 {d.get('_round', '?')} 轮）：{sentence}")

        lines += ["", "**检索角度覆盖**"]
        for sq, cov in gaps:
            lines.append(f"- [{'已覆盖' if cov else '未覆盖'}] {sq}")

        lines += ["", "**引用来源**"]
        lines += [f"- {d.get('path', '?')}" for d in docs[:5]]
        return "\n".join(lines)

    @staticmethod
    def _best_sentence(content: str, probes) -> str:
        """从文档正文中挑与查询探针重合度最高的一句话（截断到 200 字）。"""
        content = (content or "").strip()
        if not content:
            return "（无正文）"
        parts = [p.strip() for p in re.split(r"[。！？!?\n\r；;]+", content) if p.strip()]
        parts = [p for p in parts if len(p) >= 8] or [content]

        def _score(sentence: str) -> int:
            low = sentence.lower()
            return sum(1 for p in probes if p in low)

        best = max(parts, key=_score)
        return best if len(best) <= 200 else best[:200] + "…"

    def _decompose_query(self, query: str, state: AgentState = None) -> List[str]:
        """（兼容入口）把查询分解为有序子查询列表。"""
        return self._build_search_plan(query)


    def _iterative_retrieve(self, query: str, round_num: int) -> RetrievalResult:
        from fact_query import build_retrieval_result
        from rerank import rerank_documents
        # 轮内流程：先多召回 → 再 rerank 精选
        candidate_k = max(self.top_k * 3, self.top_k)
        candidates = self.corpus.search(query, candidate_k)
        docs = rerank_documents(query, candidates, top_k=self.top_k)
        result = build_retrieval_result(docs, query, round_num)
        backend = (docs[0].get("rerank_backend", "?") if docs else "-")
        print(f"    [FAISS检索] 查询「{query}」召回 {len(candidates)} 个候选，"
              f"rerank 精选出 {len(result.documents)} 个文档（后端: {backend}）")
        for doc in result.documents[:2]:
            print(f"      - [得分:{doc['score']:.4f}] {doc['path']}")
        return result

    def execute(self, state: AgentState, evaluator, router) -> AgentState:
        query = state.dialog_history[-1].user_query if state.dialog_history else ""
        self.react_steps = []
        self._searched_queries = set()
        self.ledger = EvidenceLedger()
        plan = self._build_search_plan(query)
        self.ledger.set_aspects(plan)

        print(f"\n  [Deep Search ReAct] 开始 agentic 检索: {query}")
        if self.llm.available:
            print(f"  [LLM] 可用（{self.llm.status}: {self.llm.endpoint}）")
        else:
            print(f"  [LLM] 不可用（{self.llm.reason}）→ 使用规则化 ReAct 决策")

        # 分析用户意图，提取关键槽位（用于日志与提示词）
        intent_slots = self._analyze_user_intent(query)
        self.intent_slots = intent_slots
        print(f"  [意图分析] 核心任务: {intent_slots.get('task_type', '')}")
        print(f"  [意图分析] 关键主题: {intent_slots.get('core_topics', [])}")
        print(f"  [意图分析] 过滤条件: {intent_slots.get('filters', [])}")
        print(f"  [意图分析] 输出要求: {intent_slots.get('outputs', [])}")

        for step_i in range(self.max_rounds):
            if state.elapsed_time > state.retrieval_budget:
                print(f"\n  [终止] 超过时间预算 ({state.retrieval_budget:.1f}s)")
                break

            state.increment_round()
            print(f"\n  === ReAct 第 {state.current_round} 轮 ===")

            if self.llm.available:
                action, thought, payload, meta = self._llm_react_step(query, state)
                if action in ("search", "refine", "bridge", "verify"):
                    candidate = (payload or "").strip()
                    if not candidate or self._norm_query(candidate) in self._searched_queries:
                        # LLM 给了空查询或重复查询：不允许原地打转，改用缺口驱动决策
                        print("  [LLM→规则] 查询为空或已检索过，改用证据缺口决策")
                        action, thought, payload, meta = self._rule_based_fallback(query, state)
            else:
                action, thought, payload, meta = self._rule_based_fallback(query, state)

            self.react_steps.append({
                "step": state.current_round,
                "thought": thought,
                "action": action,
                "detail": payload[:200],
                "meta": meta,
            })

            if action == "answer":
                print(f"  [Thought] {thought[:120]}")
                aspect = meta.get("aspect", "")
                print("  [Action] → answer" + (f"（方面: {aspect}）" if aspect else ""))
                if payload:
                    state.terminate(payload)
                break

            search_query = payload if payload else query
            key = self._norm_query(search_query)
            if key in self._searched_queries:
                print(f"  [去重] 「{search_query}」已检索过，本轮跳过")
                continue
            self._searched_queries.add(key)

            print(f"  [Thought] {thought[:120]}")
            aspect_hint = meta.get("aspect", "")
            if action == "bridge":
                print(f"  [Action] → bridge({aspect_hint}): {search_query}")
            elif action == "verify":
                claim_no = meta.get("claim_index")
                label = f"主张#{claim_no + 1}" if isinstance(claim_no, int) else "主张"
                print(f"  [Action] → verify({label}): {search_query}")
            elif action == "refine":
                print(f"  [Action] → refine: {search_query}")
            else:
                print(f"  [Action] → search: {search_query}")

            result = self._iterative_retrieve(search_query, state.current_round)

            # 过滤噪声文件
            filtered_docs = self._filter_noise_documents(result.documents)
            result.documents = filtered_docs

            state.add_evidence(result)

            # 台账更新：从本轮 top 文档抽取主张，并刷新方面覆盖
            probes = self._probes(query)
            for doc in result.documents[:3]:
                sent = self._best_sentence(doc.get("content", ""), probes)
                self.ledger.add_claim(
                    sent, doc.get("path", "") or "", doc.get("score", 0.0), aspect_hint
                )
            self.ledger.mark_coverage(self._plan_gaps(query, state))

            # verify：给目标主张追加首个未出现在其 sources 的命中文档作为独立来源
            if action == "verify":
                idx = meta.get("claim_index")
                if isinstance(idx, int) and 0 <= idx < len(self.ledger.claims):
                    for doc in result.documents:
                        path = doc.get("path", "") or ""
                        if path and path not in self.ledger.claims[idx].sources:
                            if self.ledger.verify(idx, path):
                                print(f"  [验证] 主张#{idx + 1} 追加独立来源: {path}")
                            break

            if evaluator:
                try:
                    done, reason = evaluator.evaluate(state, ledger=self.ledger)
                except TypeError:
                    done, reason = evaluator.evaluate(state)
                print(f"  [观察] {reason}")
                if done:
                    break
            else:
                print(f"  [观察] 已收集 {len(state.evidence)} 组证据")

        if not state.final_answer:
            if evaluator:
                try:
                    done, reason = evaluator.evaluate(state, ledger=self.ledger)
                except TypeError:
                    done, reason = evaluator.evaluate(state)
            else:
                done, reason = True, "已完成检索"
            if not done:
                gap_query, _ = self._select_gap_query(query, state)
                if gap_query:
                    print(f"\n  [补充检索] 证据仍不足（{reason}），针对缺口补充: {gap_query}")
                    state.increment_round()
                    self._searched_queries.add(self._norm_query(gap_query))
                    result = self._iterative_retrieve(gap_query, state.current_round)

                    # 过滤噪声文件
                    filtered_docs = self._filter_noise_documents(result.documents)
                    result.documents = filtered_docs

                    state.add_evidence(result)
                else:
                    print(f"\n  [终止] 证据仍不足（{reason}），且已无新的检索角度")

            # 收口：只要拿到了证据却还没有答案，就基于现有证据产出可交付的抽取式综合
            if not state.final_answer and state.evidence:
                print("  [收口] 基于现有证据生成抽取式综合回答")
                state.terminate(self._synthesize_answer(
                    query, self._plan_gaps(query, state), state, self.ledger
                ))

        return state
    
    def _analyze_user_intent(self, query: str) -> dict:
        """分析用户意图，提取关键槽位。结构化输出符合审查要求的意图。"""
        intent_slots = {
            "intent": "",
            "task_type": "",
            "core_topics": [],
            "forbidden_operations": [],
            "allowed_operations": [],
            "scope": "",
            "filters": [],
            "exclude_dirs": [],
            "exclude_exts": [],
            "exclude_prefixes": [],
            "git_filters": [],
            "content_conditions": [],
            "analysis_targets": [],
            "outputs": [],
            "skip_conditions": [],
            "need_verification": False
        }
        
        q = query.lower()
        
        # ========== 1. 核心任务识别（最高优先级） ==========
        # 识别"只读扫描 + 深度分析 + 新增报告"的核心任务
        if ("只读" in q or "不修改" in q or "禁止修改" in q) and \
           ("扫描" in q or "分析" in q) and \
           ("新增" in q or "创建" in q or "生成" in q or "review_" in q):
            intent_slots["intent"] = "只读扫描并深度分析文件，新增报告目录"
            intent_slots["task_type"] = "只读代码审查"
        elif "扫描" in q and ("分析" in q or "审计" in q):
            intent_slots["intent"] = "扫描并分析项目文件"
            intent_slots["task_type"] = "文件扫描分析"
        elif any(w in q for w in ["对比", "比较", "差异", "vs"]):
            intent_slots["intent"] = "对比分析多个主题"
            intent_slots["task_type"] = "对比分析"
        elif any(w in q for w in ["总结", "报告", "汇总"]):
            intent_slots["intent"] = "生成总结报告"
            intent_slots["task_type"] = "总结报告"
        elif any(w in q for w in ["查找", "搜索", "找"]):
            intent_slots["intent"] = "查找文件或内容"
            intent_slots["task_type"] = "文件查找"
        else:
            intent_slots["intent"] = "深度检索分析"
            intent_slots["task_type"] = "深度检索"
        
        # ========== 2. 禁止操作（最高优先级约束） ==========
        forbidden_keywords = {
            "修改": ["修改", "改", "编辑", "更新", "覆盖", "overwrite"],
            "移动": ["移动", "move"],
            "删除": ["删除", "移除", "清理", "删掉", "干掉", "remove", "delete"],
            "重命名": ["重命名", "rename"],
            "覆盖": ["覆盖", "覆写", "overwrite"]
        }
        for op, keywords in forbidden_keywords.items():
            if any(kw in q for kw in keywords):
                intent_slots["forbidden_operations"].append(op)
        
        # 如果明确说"不修改"或"只读"，添加所有禁止操作
        if "只读" in q or "不修改" in q or "禁止修改" in q or "不要修改" in q:
            intent_slots["forbidden_operations"] = ["修改", "移动", "删除", "重命名", "覆盖"]
        
        # ========== 3. 允许操作 ==========
        intent_slots["allowed_operations"] = ["读取", "查询", "分析"]
        if "新增" in q or "创建" in q or "生成" in q or "review_" in q:
            intent_slots["allowed_operations"].append("新增 review_* 目录和报告")
        
        # ========== 4. 作用域 ==========
        if "当前项目" in q or "整个项目" in q:
            intent_slots["scope"] = "当前项目及一级子项目中的文本文件"
        else:
            intent_slots["scope"] = "当前项目中的文件"
        
        # ========== 5. 排除目录 ==========
        exclude_dir_patterns = [
            (r'\.git\b', '.git'),
            (r'node_modules', 'node_modules'),
            (r'\bdist\b', 'dist'),
            (r'\bbuild\b', 'build'),
            (r'\bbackup\b', 'backup'),
            (r'\barchive\b', 'archive'),
            (r'\bvendor\b', 'vendor'),
            (r'__pycache__', '__pycache__'),
            (r'\.venv\b', '.venv'),
        ]
        for pattern, dirname in exclude_dir_patterns:
            if re.search(pattern, q):
                intent_slots["exclude_dirs"].append(dirname)
        
        # ========== 6. 排除扩展名 ==========
        ext_pattern = r'\.(tmp|log|lock|min\.js|map)\b'
        ext_matches = re.findall(ext_pattern, q)
        for ext in ext_matches:
            intent_slots["exclude_exts"].append(f".{ext}")
        
        # ========== 7. 排除前缀 ==========
        if "_draft" in q:
            intent_slots["exclude_prefixes"].append("_draft")
        
        # ========== 8. Git 过滤 ==========
        if ".gitignore" in q:
            intent_slots["git_filters"].append(".gitignore")
        if ".git/info/exclude" in q:
            intent_slots["git_filters"].append(".git/info/exclude")
        if "git log" in q or "git" in q:
            intent_slots["git_filters"].append("最近30天 git log")
        if "ci-bot" in q or "dependabot" in q:
            intent_slots["git_filters"].append("作者不是 ci-bot/dependabot")
        
        # ========== 9. 内容条件 ==========
        content_conditions = []
        if re.search(r'TODO|FIXME|XXX|HACK', q, re.IGNORECASE):
            content_conditions.append("未注释 TODO/FIXME/XXX/HACK")
        if "import" in q or "require" in q or "from" in q or "依赖" in q:
            content_conditions.append("import/require/from 依赖")
        if re.search(r'相似度.*0\.82|0\.82.*相似度', q):
            content_conditions.append("相似度 > 0.82")
        if re.search(r'复杂度.*15|15.*复杂度', q):
            content_conditions.append("复杂度 > 15")
        intent_slots["content_conditions"] = content_conditions
        
        # ========== 10. 分析目标 ==========
        analysis_targets = []
        if "缺失文档" in q or "文档" in q:
            analysis_targets.append("缺失文档")
        if "循环依赖" in q:
            analysis_targets.append("循环依赖")
        if "孤儿" in q:
            analysis_targets.append("孤儿模块")
        if "未使用" in q or "导出" in q:
            analysis_targets.append("未使用导出")
        if "异常" in q:
            analysis_targets.append("未处理异常")
        if "TODO" in q:
            analysis_targets.append("TODO 优先级")
        if "公开 API" in q or "API" in q:
            analysis_targets.append("公开 API")
        if "配置键" in q:
            analysis_targets.append("配置键")
        if "环境变量" in q:
            analysis_targets.append("环境变量")
        if "URL" in q or "url" in q:
            analysis_targets.append("硬编码 URL")
        if "密钥" in q or "key" in q.lower():
            analysis_targets.append("密钥疑似项")
        if "测试" in q:
            analysis_targets.append("测试覆盖引用")
        if not analysis_targets:
            analysis_targets = ["缺失文档", "循环依赖", "孤儿模块", "未使用导出", "未处理异常"]
        intent_slots["analysis_targets"] = analysis_targets
        
        # ========== 11. 输出要求 ==========
        outputs = []
        if "review_" in q:
            outputs.append("review_YYYYMMDD_HHmmss 目录")
        if "Markdown" in q or ".md" in q or "总报告" in q:
            outputs.append("Markdown 总报告")
        if "JSON" in q or ".json" in q or "索引" in q:
            outputs.append("JSON 机器可读索引")
        if "CSV" in q or ".csv" in q or "明细" in q:
            outputs.append("CSV 明细表")
        if "摘要" in q or "分析摘要" in q:
            outputs.append("按原相对路径命名的只读分析摘要文件")
        if "跳过" in q and "原因" in q:
            outputs.append("跳过原因记录")
        if "校验" in q or "验证" in q:
            outputs.append("原文件未改动校验")
        if not outputs:
            outputs = ["review_YYYYMMDD_HHmmss 目录", "Markdown 总报告", "JSON 索引", "CSV 明细"]
        intent_slots["outputs"] = outputs
        
        # ========== 12. 跳过条件 ==========
        skip_conditions = []
        skip_keywords = ["符号链接", "只读文件", "权限不足", "二进制", "编码无法识别", ">5MB", "超过 5MB"]
        for kw in skip_keywords:
            if kw in q:
                skip_conditions.append(kw)
        if not skip_conditions:
            skip_conditions = ["符号链接", "只读文件", "权限不足", "二进制", "编码无法识别", ">5MB"]
        intent_slots["skip_conditions"] = skip_conditions
        
        # ========== 13. 验证要求 ==========
        if "校验" in q or "验证" in q or "不修改原文件" in q:
            intent_slots["need_verification"] = True

        # ========== 14. 汇总字段（日志与提示词共用，保证键始终存在） ==========
        intent_slots["filters"] = (
            [f"排除目录 {d}" for d in intent_slots["exclude_dirs"]]
            + [f"排除扩展名 {e}" for e in intent_slots["exclude_exts"]]
            + [f"排除前缀 {p}" for p in intent_slots["exclude_prefixes"]]
            + [f"git 过滤 {g}" for g in intent_slots["git_filters"]]
        )
        try:
            _, entities, _ = self.query_rewriter.rewrite(query)
        except Exception:
            entities = []
        topic = self._topic_of(query)
        intent_slots["core_topics"] = list(dict.fromkeys(
            [t for t in [topic] + list(entities) if t]
        ))

        return intent_slots
    
    def _extract_core_query(self, query: str) -> str:
        """从查询中提取核心主题。"""
        # 移除阈值参数
        query = re.sub(r'相似度[超过>=]+\s*\d+\.?\d*', '', query)
        query = re.sub(r'复杂度[超过>=]+\s*\d+\.?\d*', '', query)
        query = re.sub(r'\d+\.?\d*\s*[或和]\s*\d+\.?\d*', '', query)
        
        # 移除过滤扩展名
        query = re.sub(r'\.[a-z]+/[.\w/]+', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\.\w+(?=[\s,，]|$)', '', query)
        
        # 移除目录排除条件
        query = re.sub(r'排除?\s*[\w./]+目录', '', query)
        query = re.sub(r'除\s*[\w./]+\s*之外', '', query)
        
        # 清理多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query if query else "项目分析"
    
    def _filter_noise_documents(self, documents: List[Dict]) -> List[Dict]:
        """过滤噪声文件，保留高质量文档。"""
        noise_patterns = ['.operation_log.json', '.log', '.tmp', '.lock', '.min.js', '.map']
        
        filtered = []
        for doc in documents:
            path = doc.get("path", "").lower()
            
            # 检查是否是噪声文件
            is_noise = any(noise in path for noise in noise_patterns)
            
            # 检查得分是否足够高
            score = doc.get("score", 0.0)
            
            # 如果是噪声文件但得分很高，仍然保留但降低优先级
            if is_noise and score > 0.5:
                doc["score"] = score * 0.3  # 降低噪声文件的得分
                filtered.append(doc)
            elif not is_noise:
                filtered.append(doc)
        
        # 按得分排序，保留前5个
        filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return filtered[:5]

    def _llm_react_step(self, query: str, state: AgentState
                        ) -> Tuple[str, str, str, Dict[str, Any]]:
        """执行一步 LLM ReAct 决策。"""
        system = REACT_SYSTEM_PROMPT.format(
            round_num=state.current_round,
            max_rounds=self.max_rounds,
            doc_count=sum(len(ev.documents) for ev in state.evidence),
        )
        user = self._build_user_prompt(query, state)

        llm_output = self.llm.chat(system, user)
        if llm_output:
            print(f"  [LLM输出] {llm_output[:150]}...")
            return self._parse_react_action(llm_output)

        print("  [LLM] 无输出，回退到规则决策")
        action, thought, payload, meta = self._rule_based_fallback(query, state)
        return action, f"{thought}（LLM 无输出）", payload, meta

    def get_corpus_stats(self) -> Dict:
        return self.corpus.stats()
