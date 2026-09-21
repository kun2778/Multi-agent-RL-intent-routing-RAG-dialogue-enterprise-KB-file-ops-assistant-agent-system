from typing import List, Dict, Any
from agent_state import AgentState


class ClarificationHandler:
    """
    Vague/Incomplete Branch: Active clarification using IntentRL.
    Pauses retrieval when key elements are missing and generates clarification questions.
    """

    def __init__(self):
        self.max_clarifications = 3
        self.pending_clarifications: List[str] = []

    def generate_clarification(self, query: str) -> List[str]:
        """Generate clarification questions for incomplete queries."""
        from query_rewriter import QueryRewriter
        rewriter = QueryRewriter()
        return rewriter.generate_clarification(query)

    def handle(self, state: AgentState, clarifications: List[str]) -> AgentState:
        """
        Handle vague query: pause retrieval, generate questions, wait for user.
        Returns state with clarification questions.
        """
        state.clarifications = clarifications[:self.max_clarifications]

        print(f"\n{'='*60}")
        print(f"  [主动澄清] 用户问题缺少关键要素，暂停检索。")
        print(f"  请回答以下问题以补充信息：")
        for i, q in enumerate(state.clarifications, 1):
            print(f"    Q{i}: {q}")
        print(f"{'='*60}")

        return state

    def resolve(self, state: AgentState, user_answer: str) -> AgentState:
        """
        Resolve clarification with user's answer.
        Forms a closed loop back to intent recognition.
        """
        print(f"\n>>> [闭环] 用户补充信息: '{user_answer}'")
        state.add_dialog(user_answer, "")
        state.clarifications = []

        return state

    def is_clarification_pending(self, state: AgentState) -> bool:
        """Check if clarification is still pending."""
        return len(state.clarifications) > 0 and not state.is_terminated
