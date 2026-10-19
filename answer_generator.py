import os
from typing import List
from agent_state import AgentState, IntentType

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMGenerator:
    """真实 LLM 答案生成器（带引用要求），无 key 时回退为规则合成。"""

    def __init__(self):
        self.client = None
        self._setup_client()

    def _setup_client(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def generate_with_llm(self, context: str, query: str) -> str:
        if not self.client:
            return self._fallback_generate(context, query)
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": (
                        "你是专业的深度搜索助手。基于证据上下文回答问题。\n"
                        "要求：1. 必须引用来源，格式：(来源：文件路径)\n"
                        "2. 证据不足时明确说明\n3. 逻辑清晰、可溯源"
                    )},
                    {"role": "user", "content": f"问题：{query}\n\n证据上下文：\n{context}\n\n请生成带引用的答案："},
                ],
                max_tokens=500, temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [LLM] 生成失败: {e}，使用回退方案")
            return self._fallback_generate(context, query)

    def _fallback_generate(self, context: str, query: str) -> str:
        lines = ["### 回答", f"基于检索到的证据，对问题「{query}」的回答如下：\n"]
        for i, item in enumerate(context.split("[证据")[1:], 1):
            lines.append(f"{i}. {item.strip()[:200]}")
            lines.append("")
        lines.append("### 结论")
        lines.append("以上基于多轮检索证据综合回答，来源均已标注。")
        return "\n".join(lines)


OP_INTENTS = {IntentType.ADD_FILE, IntentType.DELETE_FILE,
              IntentType.MODIFY_FILE, IntentType.REFACTOR}


class AnswerGenerator:
    """答案生成：操作类任务输出操作总结，检索类任务输出带引用的深度答案。"""

    def __init__(self):
        self.llm_generator = LLMGenerator()

    def generate(self, state: AgentState) -> str:
        if state.last_operation() and state.current_intent in OP_INTENTS:
            return self._format(self._operation_answer(state), state)
        if state.final_answer:
            return self._format(state.final_answer, state)

        query = state.dialog_history[0].user_query if state.dialog_history else ""
        context = state.build_llm_context()
        result = self._build_search_answer(query, context, state)
        return self._format(result, state)

    def _operation_answer(self, state: AgentState) -> str:
        ev = state.last_operation()
        status = "成功" if ev.success else "失败"
        lines = [
            f"### 操作结果: {status}",
            f"- 操作类型: {ev.kind}",
            f"- 目标文件: {ev.target or '（未指定）'}",
            f"- 详情: {ev.message}",
        ]
        if getattr(state, 'last_outcome', ''):
            lines.append(f"- RL事件: {state.last_outcome}")
        return "\n".join(lines)

    def _build_search_answer(self, query: str, context: str, state: AgentState) -> str:
        lines = [f"### 用户问题: {query}\n"]
        if state.routing_path and state.routing_path.value == "deep_search":
            lines.append("### 分析结果")
        else:
            lines.append("### 查询结果")

        total_docs = sum(len(ev.documents) for ev in state.evidence)
        lines.append(f"共收集 {len(state.evidence)} 轮检索结果，涵盖 {total_docs} 个文档块。\n")
        lines.append("### 详细回答:")
        lines.append(self.llm_generator.generate_with_llm(context, query))
        lines.append("\n### 引用来源:")
        lines.extend(self._generate_citations(state))
        return "\n".join(lines)

    def _generate_citations(self, state: AgentState) -> List[str]:
        citations = []
        seen = set()
        for ev in state.evidence:
            for doc in ev.documents:
                path = doc.get("path", doc.get("source", "未知"))
                if path in seen:
                    continue
                seen.add(path)
                citations.append(f"  - 《{path}》")
        return citations if citations else ["  无引用来源"]

    def _format(self, answer: str, state: AgentState) -> str:
        intent = state.current_intent.value if state.current_intent else 'N/A'
        ops = state.operation_history
        op_summary = ", ".join(f"{e.kind}[{'✓' if e.success else '✗'}]" for e in ops) or "无"
        return f"""
{'='*70}
  [结果已生成] AgentRL + Deep Search 对话框系统
{'='*70}

{answer}

{'='*70}
  运行统计
{'='*70}
  回合/最大: {state.current_round}/{state.max_rounds}
  意图: {intent}    路径: {state.routing_path.value if state.routing_path else 'N/A'}
  语料库文件: {state.corpus_size}     操作: {op_summary}
  检索耗时: {state.elapsed_time:.1f}s
  RL奖励事件: {len(state.rl_rewards)}
{'='*70}
"""