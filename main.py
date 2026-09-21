import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from agent_state import AgentState
from agent_rl import AgentRL
from marl_agent import MetaAgent
from router import Router
from answer_generator import AnswerGenerator
from file_corpus import FileCorpus


class AgentRLSystem:
    """
    对话框驱动的 AgentRL + Deep Search 系统。

    用户在对话框中输入自然语言，系统通过 AgentRL 识别操作意图（添加/删除/
    修改/整理/查询/深度检索/澄清），执行真实文件操作或深度检索，并按终止
    条件模块结束回合。语料库为可增删改的真实文件（workspace/）。

    支持两种Agent架构：
      - single: 单Agent架构（原版）
      - marl: 多Agent协作架构（新版，推荐）
    """

    def __init__(self, agent_type: str = None):
        """
        Initialize AgentRLSystem.

        Args:
            agent_type: "single" or "marl". If None, reads from AGENT_TYPE env var,
                       defaults to "marl".
        """
        if agent_type is None:
            agent_type = os.environ.get("AGENT_TYPE", "marl").lower()

        self.corpus = FileCorpus(use_faiss_store=True)
        self.corpus.seed_demo()
        self.router = Router(corpus=self.corpus, agent_type=agent_type)
        self.generator = AnswerGenerator()
        self.agent_rl = self.router.agent_rl
        self.agent_type = agent_type

    def process_query(self, query: str) -> str:
        """处理单条对话框输入，返回最终结果字符串。"""
        query = query.strip()
        if not query:
            return ""

        state = AgentState()
        state.corpus_size = len(self.corpus.registry)
        state.add_dialog(query, "")

        print(f"\n{'#'*70}")
        print(f"  [用户输入] {query}")
        print(f"{'#'*70}")

        state = self.router.route(state)

        while state.clarifications and not state.is_terminated:
            print("\n  [等待澄清回答] 请输入补充信息：")
            user_answer = input("  > ").strip()
            if not user_answer:
                print("  空输入，跳过澄清，任务结束。")
                state.terminate("未提供澄清信息。")
                break
            state = self.router.resolve_clarification(state, user_answer)

        answer = self.generator.generate(state)
        state.final_answer = state.final_answer or answer
        print("\n" + "=" * 62)
        print("  [结果]")
        print("=" * 62)
        print(answer)
        print("\n" + state.summary())
        self.corpus._save_registry()
        return answer

    # ------------------------------------------------------------- 交互循环
    def interactive(self):
        print("\n" + "=" * 70)
        print("  AgentRL + Deep Search 对话框系统")
        print("  输入自然语言即可：添加/删除/修改文件、整理重构、查询、深度分析")
        print("  - 输入「帮助」查看指令  -「列出」查看语料库  -「退出/exit」结束")
        print("  - 语料库目录: " + self.corpus.root)
        print("=" * 70)

        print("\n  [启动] 加载持久化模型/预热训练...")
        if self.agent_rl.loaded_from_disk:
            print("  [启动] 存在已训练模型，跳过预热训练（在线学习中持续更新）。")
        else:
            print("  [启动] 首次运行：执行预热 RL 训练并保存模型（下次启动自动加载）...")
            self.agent_rl.train_episode(num_episodes=8)
            self.agent_rl.save()

        while True:
            try:
                raw = input("\n 你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not raw:
                continue

            lower = raw.lower()
            if lower in ("exit", "quit", "退出", "bye", "再见", "q"):
                print("再见！")
                break
            if self._is_command(raw):
                self._run_command(raw)
                continue

            self.process_query(raw)

    def _is_command(self, raw: str) -> bool:
        for prefix in ("帮助", "help", "列出", "list", "ls", "查看", "view", "cat",
                       "训练", "train", "演示", "demo", "状态", "status", "语料",
                       "切换", "switch", "比较", "compare"):
            if raw.startswith(prefix) or raw.lower().startswith(prefix.lower()):
                return True
        return False

    def _run_command(self, raw: str):
        lower = raw.lower()
        if lower.startswith("帮助") or lower.startswith("help"):
            self._print_help()
        elif lower.startswith("列出") or lower.startswith("list") or lower.startswith("ls"):
            self._list_documents()
        elif lower.startswith("查看") or lower.startswith("view") or lower.startswith("cat"):
            target = raw.split(None, 1)[1].strip() if len(raw.split()) > 1 else ""
            self._view_document(target)
        elif lower.startswith("训练") or lower.startswith("train"):
            self.agent_rl.train_episode(num_episodes=5)
        elif lower.startswith("演示") or lower.startswith("demo"):
            self._run_demo()
        elif lower.startswith("状态") or lower.startswith("status"):
            self._print_status()
        elif lower.startswith("切换") or lower.startswith("switch"):
            self._switch_agent(raw)
        elif lower.startswith("比较") or lower.startswith("compare"):
            self._compare_agents(raw)
        else:
            self._print_help()

    def _list_documents(self):
        docs = self.corpus.list_documents()
        print(f"\n[语料库] 共 {len(docs)} 个文件（{self.corpus.root}）:")
        for d in docs:
            flag = "可读" if d["readable"] else "二进制"
            tags = ",".join(d["tags"]) if d["tags"] else "-"
            print(f"  • {d['path']}  ({d['type']}, {d['size']}B, {flag}, 标签:{tags})")

    def _view_document(self, target: str):
        if not target:
            print("用法: view <文件名>")
            return
        content = self.corpus.read_text(target)
        matched = self.corpus._resolve_target(target)
        if content is None:
            print(f"  [✗] 无法读取 {target}（未找到或为二进制文件）" if matched else f"  [✗] 未找到 {target}")
            return
        print(f"\n[{matched}]\n{content}")

    def _print_status(self):
        """Print detailed status of the current agent."""
        print("\n" + "=" * 60)
        print("  系统状态")
        print("=" * 60)

        agent_class = self.agent_rl.__class__.__name__
        print(f"  Agent类型: {agent_class} ({self.agent_type})")
        print(f"  Epsilon: {self.agent_rl.epsilon:.4f}")

        # Handle both single and MARL agent buffers
        if hasattr(self.agent_rl, 'replay_buffer'):
            buffer = self.agent_rl.replay_buffer
            if hasattr(buffer, '__len__'):
                print(f"  经验池: {len(buffer)} 条")
            elif hasattr(buffer, '_size'):
                print(f"  经验池: {buffer._size} 条")

        print(f"  训练步数: {self.agent_rl.training_step}")

        # Show MARL-specific info
        if self.agent_type == "marl" and isinstance(self.agent_rl, MetaAgent):
            print("\n  [MARL 多Agent架构]")
            print(f"    IntentAgent Q网络参数: {sum(p.numel() for p in self.agent_rl.intent_agent.q_network.parameters())}")
            print(f"    SubIntentAgent Q网络参数: {sum(p.numel() for p in self.agent_rl.subintent_agent.q_network.parameters())}")
            print(f"    SlotAgent 网络参数: {sum(p.numel() for p in self.agent_rl.slot_agent.network.parameters())}")

        print(f"\n  语料库: {str(self.corpus.stats())}")
        print("=" * 60)

    def _switch_agent(self, raw: str):
        """Switch between single-agent and MARL architectures."""
        parts = raw.split()
        if len(parts) < 2:
            print("\n用法: 切换 <类型>")
            print("  类型: single - 单Agent架构")
            print("        marl   - 多Agent协作架构（推荐）")
            print(f"\n当前: {self.agent_type}")
            return

        new_type = parts[1].lower()
        if new_type not in ("single", "marl"):
            print(f"  [✗] 未知类型: {new_type}，请使用 'single' 或 'marl'")
            return

        if new_type == self.agent_type:
            print(f"  [✓] 已经是 {new_type} 架构")
            return

        self.router.switch_agent(new_type)
        self.agent_rl = self.router.agent_rl
        self.agent_type = new_type
        print(f"  [✓] 已切换到 {new_type} 架构")

    def _compare_agents(self, raw: str):
        """Compare predictions from both agent architectures."""
        parts = raw.split(None, 1)
        if len(parts) < 2:
            print("\n用法: 比较 <查询>")
            print("  示例: 比较 帮我把最近的截屏内容的文字进行识别并输出")
            return

        query = parts[1].strip()
        if not query:
            print("  [✗] 请输入查询内容")
            return

        print(f"\n{'='*60}")
        print(f"  比较两种Agent架构的预测结果")
        print(f"  查询: {query}")
        print(f"{'='*60}")

        results = self.router.compare_agents(query)

        print("\n[单Agent架构]")
        single = results.get("single_agent", {})
        if "error" in single:
            print(f"  错误: {single['error']}")
        else:
            print(f"  意图: {single.get('intent', 'N/A')}")
            print(f"  路径: {single.get('path', 'N/A')}")
            print(f"  置信度: {single.get('confidence', 0):.4f}")

        print("\n[MARL多Agent架构]")
        marl = results.get("marl_agent", {})
        if "error" in marl:
            print(f"  错误: {marl['error']}")
        else:
            print(f"  意图: {marl.get('intent', 'N/A')}")
            print(f"  路径: {marl.get('path', 'N/A')}")
            print(f"  置信度: {marl.get('confidence', 0):.4f}")

        if "match" in results:
            match_str = "✓ 一致" if results["match"] else "✗ 不一致"
            print(f"\n[比较结果] {match_str}")

        print("=" * 60)

    def _print_help(self):
        print("""
用途说明（输入自然语言）:
  添加/新建  : 添加文件 notes.txt 内容为 你好              → 写入新文件
  删除/移除  : 删除文件 notes.txt                          → 移入 .trash
  修改/更新  : 修改文件 config.yaml 内容为 ...            → 备份后改写
  整理/重构  : 整理文件夹                                  → 按类型归类
  查询/检索  : 搜索 强化学习                              → 单轮向量检索
  分析/对比  : 对比分析 config.yaml 和 report.md           → 多轮深度检索

命令:
  help/帮助      列出/列表/list    查看/view <文件>
  训练/train     演示/demo         状态/status
  切换/switch    比较/compare      退出/exit

Agent架构:
  切换 single    → 单Agent架构（原版）
  切换 marl      → 多Agent协作架构（推荐，更准确）
  比较 <查询>    → 对比两种架构的预测结果
""")

    def _run_demo(self):
        self.agent_rl.train_episode(num_episodes=3)
        demo_inputs = [
            "添加文件 demo_notes.txt 内容为 强化学习通过奖励信号训练策略",
            "对比分析 强化学习.txt 和 demo_notes.txt",
            "整理文件夹",
            "搜索 强化学习",
            "删除文件 demo_notes.txt",
        ]
        for q in demo_inputs:
            print(f"\n\n{'*'*70}\n  [演示] {q}\n{'*'*70}")
            self.process_query(q)


def main():
    system = AgentRLSystem()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        system.process_query(query)
    else:
        system.interactive()


if __name__ == "__main__":
    main()