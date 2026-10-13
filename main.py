import os
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from agent_state import AgentState
from agent_rl import AgentRL
from marl_agent import MetaAgent
from router import Router
from answer_generator import AnswerGenerator
from file_corpus import FileCorpus
from continuous_learning import get_learning_engine


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
        self.learning_engine = get_learning_engine()
        self._replay_ingest()

    def _replay_ingest(self):
        """启动时把线上 bad case 脱敏后回流进 replay buffer。

        失败已由 continuous_learning 落盘到 learning/failures/；
        这里补一次回流（幂等，已回流过的按 sha1 记账跳过）。
        回流失败只打日志，绝不阻塞系统启动。
        """
        try:
            from badcase_replay import push_bad_cases, replay_status
            pushed, skipped = push_bad_cases(self.agent_rl)
            pending = replay_status().get("bad_cases_pending", 0)
        except Exception as exc:
            print(f"  [回流] bad case 回流失败（不影响启动）: {exc}")
            return
        if pushed:
            print(f"  [回流] bad case → replay buffer: +{pushed} 条 "
                  f"(跳过 {skipped}，buffer={len(self.agent_rl.replay_buffer)})")
        elif pending:
            # 显式暴露：以前这类情况被逐条 except 静默吞掉，看起来像「回流正常」
            print(f"  [回流] 有 {pending} 条待回流样本未写入 buffer"
                  f"（已跳过 {skipped} 条，当前架构={self.agent_type}）")

    def process_query(self, query: str) -> str:
        """处理单条对话框输入，返回最终结果字符串。"""
        query = query.strip()
        if not query:
            return ""

        start_time = time.time()

        state = AgentState()
        state.corpus_size = len(self.corpus.registry)
        state.add_dialog(query, "")

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

        latency_ms = (time.time() - start_time) * 1000

        # 记录 Trace
        retrieval_docs = []
        retrieval_scores = []
        for ev in state.evidence:
            for d in ev.documents:
                retrieval_docs.append(d.get("path", ""))
            if ev.score:
                retrieval_scores.append(ev.score)

        thinking = self._build_thinking(state)
        trace_id = self.learning_engine.record_trace(
            user_query=query,
            answer=answer,
            intent=state.current_intent.value if state.current_intent else "",
            route=state.routing_path.value if state.routing_path else "",
            thinking=thinking,
            clarifications=list(state.clarifications or []),
            retrieval_docs=retrieval_docs,
            retrieval_scores=retrieval_scores,
            latency_ms=round(latency_ms, 2),
            agent=self.agent_type,
        )
        self.learning_engine.increment_trace_count()

        print("\n" + "=" * 62)
        print("  [结果]")
        print("=" * 62)
        print(answer)
        print("\n" + state.summary())
        print(f"\n  [Trace ID] {trace_id}")

        # 收集用户反馈
        self._collect_feedback(trace_id)

        self.corpus._save_registry()
        return answer

    def _build_thinking(self, state: AgentState) -> list:
        """从 state 构建 thinking steps 列表。"""
        from agent_state import IntentType, SHORT_LABELS
        steps = []
        if state.current_intent is not None:
            label = SHORT_LABELS.get(state.current_intent, state.current_intent.value)
            path = state.routing_path.value if state.routing_path else "N/A"
            steps.append({
                "title": "意图识别",
                "detail": f"用户请求被分为【{label}】类 ({state.current_intent.value})",
                "extra": f"路由路径 {path}",
            })
        for ev in state.operation_history:
            steps.append({
                "title": "文件操作",
                "detail": f"操作: {ev.kind} 文件「{ev.target or '（未指定）'}」",
                "extra": f"结果[{'成功' if ev.success else '失败'}]: {ev.message}",
            })
        for r in state.evidence:
            files = [d.get("path") for d in r.documents if d.get("path")]
            steps.append({
                "title": "文档检索",
                "detail": f"第 {r.round_num} 轮向量检索命中 {len(files)} 个文件",
                "extra": "、".join(files) if files else "无命中文档",
            })
        if state.clarifications:
            steps.append({
                "title": "主动澄清",
                "detail": "指令存在歧义，需要用户补充信息后才能继续",
                "extra": "；".join(state.clarifications),
            })
        if not steps:
            steps.append({"title": "已接收", "detail": "请求已处理完成", "extra": ""})
        return steps

    def _collect_feedback(self, trace_id: str):
        """命令行交互：收集用户 T/F 反馈。"""
        try:
            feedback_input = input("\n  这个回答有帮助吗？(T=有帮助 / F=无帮助 / 回车跳过): ").strip().lower()
            if feedback_input == 't':
                self.learning_engine.update_feedback(trace_id, "positive")
                print("  ✓ 已记录为「有帮助」")
            elif feedback_input == 'f':
                self.learning_engine.update_feedback(trace_id, "negative")
                print("  ✓ 已记录为「无帮助」，感谢反馈！")

                # 进一步归因
                print("  失败类型：")
                print("    1. 意图分类错  2. 路由错  3. 检索错")
                print("    4. 生成错  5. 策略错  6. 澄清错  7. 操作错  8. 跳过")
                type_input = input("  请选择 (1-8): ").strip()
                type_map = {
                    "1": "intent_classification_error",
                    "2": "route_error",
                    "3": "retrieval_error",
                    "4": "generation_error",
                    "5": "strategy_error",
                    "6": "clarification_error",
                    "7": "operation_error",
                }
                if type_input in type_map:
                    from continuous_learning import TRACES_DIR
                    trace_path = TRACES_DIR / f"{trace_id}.json"
                    if trace_path.exists():
                        import json
                        with open(trace_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data["human_label"] = {
                            "failure_type": type_map[type_input],
                            "notes": "命令行用户标注",
                        }
                        from atomic_io import atomic_write_json
                        atomic_write_json(trace_path, data)
                        print(f"  ✓ 已记录失败类型: {type_map[type_input]}")
            # feedback_input == '' -> 跳过
        except (EOFError, KeyboardInterrupt):
            pass

    # ------------------------------------------------------------- 交互循环
    def interactive(self):
        print("\n" + "=" * 70)
        print("  AgentRL + Deep Search 对话框系统")
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
                       "切换", "switch", "比较", "compare", "学习报告", "learn"):
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
        elif lower.startswith("学习报告") or lower.startswith("learn"):
            self._print_learning_report()
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
  切换/switch    比较/compare      学习报告/learn
  退出/exit

Agent架构:
  切换 single    → 单Agent架构（原版）
  切换 marl      → 多Agent协作架构（推荐，更准确）
  比较 <查询>    → 对比两种架构的预测结果

持续学习:
  每次回答后输入 T/F 标记反馈
  输入「学习报告」查看失败统计和归因分布
""")

    def _print_learning_report(self):
        """打印持续学习统计报告。"""
        engine = self.learning_engine
        stats = engine.get_stats()
        failures = engine.get_failure_cases(limit=100)
        versions = engine.list_versions()

        print("\n" + "=" * 60)
        print("  持续学习报告")
        print("=" * 60)

        total = stats.get("total_traces", 0)
        fail_count = stats.get("total_failures", 0)
        fail_rate = (fail_count / total * 100) if total > 0 else 0
        print(f"  总 Trace 数: {total}")
        print(f"  失败案例数: {fail_count}")
        print(f"  失败率: {fail_rate:.1f}%")

        by_type = stats.get("by_type", {})
        if by_type:
            print(f"\n  失败归因分布:")
            type_labels = {
                "intent_classification_error": "意图分类错",
                "route_error": "路由错",
                "retrieval_error": "检索错",
                "generation_error": "生成错",
                "strategy_error": "策略错",
                "clarification_error": "澄清错",
                "operation_error": "操作错",
                "unknown": "未知",
            }
            for ftype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                label = type_labels.get(ftype, ftype)
                bar = "█" * min(count, 30)
                print(f"    {label:8s}  {count:3d}  {bar}")

        if versions:
            print(f"\n  版本历史: {len(versions)} 个")
            for v in versions[-3:]:
                print(f"    {v['version_id']}  [{v['status']}]")

        if failures:
            print(f"\n  最近失败案例 (共 {len(failures)} 个):")
            for f in failures[:5]:
                query = f.get("user_query", "")[:40]
                attr = f.get("attribution", {})
                ftype = type_labels.get(attr.get("primary_failure", ""), "未知")
                cause = attr.get("root_cause", "")[:30]
                print(f"    • [{ftype}] {query}... → {cause}")

        print("=" * 60)

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