"""长期高效训练脚本 v2：蒸馏 + 分类预训练 + DQN 强化。

用法:
  python train_strong.py [steps] [log_every] [--fresh] [--eval-only] [--distill] [--csv]
    steps         训练迭代次数（默认 200000）
    log_every     每多少步打印并保存（默认 5000）
    --fresh       删除旧模型重新训练
    --eval-only   仅评估（默认续训）
    --distill     先用蒸馏标签预训练分类头，再 RL 微调
    --csv         用 datasets/intent_corpus.csv 训练分类头（推荐先跑）
"""

import random
import sys
import time
import os

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from agent_rl import AgentRL, DEFAULT_MODEL_PATH
from agent_state import AgentState, IntentType


LLM_DISTILL_LABELS = {
    "你好，你是谁啊": "vague",
    "今天天气怎么样": "vague",
    "帮我一下": "vague",
    "你能做什么": "vague",
    "讲个笑话": "vague",
    "随便聊聊": "vague",
    "谢谢你的帮助": "vague",
    "嗯嗯好的": "vague",
    "我是谁": "vague",
    "hello": "vague",
    "帮我新建一个 readme.md 文件": "add_file",
    "创建文件 notes.txt 内容是学习笔记": "add_file",
    "写入 data.csv 第一行是 id,name,score": "add_file",
    "新建 main.py 写一个 hello world": "add_file",
    "把这段内容保存到 output.txt": "add_file",
    "帮我写一个 test.py": "add_file",
    "生成 config.json 配置文件": "add_file",
    "添加一个 requirements.txt": "add_file",
    "删除文件 backup.py": "delete_file",
    "移除 temp 目录下的所有文件": "delete_file",
    "把那个旧的 log.txt 删了": "delete_file",
    "清理掉过期的 cache 文件": "delete_file",
    "移除掉那个没用的脚本": "delete_file",
    "删掉 data 目录": "delete_file",
    "修改 config.yaml 把端口改成 8080": "modify_file",
    "更新 main.py 的导入语句": "modify_file",
    "编辑 README 把标题改一下": "modify_file",
    "帮我改一下 config.json 的超时时间": "modify_file",
    "把 settings.py 里的 debug 改成 True": "modify_file",
    "修改 app.js 的路由配置": "modify_file",
    "整理一下 src 文件夹": "refactor",
    "重构项目结构按模块分目录": "refactor",
    "把文件按类型分类归档": "refactor",
    "重命名 main.py 为 app.py": "refactor",
    "移动测试文件到 tests 目录": "refactor",
    "代码重构优化目录结构": "refactor",
    "帮我整理一下代码": "refactor",
    "搜索强化学习": "query",
    "查询 config.yaml 的内容": "query",
    "找到红警3的启动程序在哪里": "query",
    "这个项目用了什么框架": "query",
    "查看 main.py 的代码": "query",
    "查找文件里的关键词": "query",
    "对比分析强化学习和深度学习": "deep_search",
    "分析 config.yaml 和 settings.yaml 的差异": "deep_search",
    "深度分析项目的性能瓶颈": "deep_search",
    "综合分析各个模块的优缺点": "deep_search",
    "评估不同方案的利弊": "deep_search",
    "帮我比较两个版本的区别": "deep_search",
    "研究一下最新的技术趋势": "deep_search",
}


def eval_accuracy(agent, rounds: int = 40) -> dict:
    """模板准确率（有偏上限）。"""
    acc = {}
    with torch.no_grad():
        for task, templates in agent.TASK_TEMPLATES.items():
            ok = 0
            for _ in range(rounds):
                f = random.choice(agent.SAMPLE_FILES)
                g = random.choice([x for x in agent.SAMPLE_FILES if x != f])
                query = random.choice(templates).format(f=f, g=g, c=agent.SAMPLE_CONTENT)
                stub = AgentState()
                stub.increment_round()
                vec = agent._extract_features(query, stub)
                cls_probs = agent._classify_probs(vec)
                q_vals = agent.q_network(torch.FloatTensor(vec).unsqueeze(0))
                q_probs = torch.softmax(q_vals, dim=-1).squeeze(0).numpy()
                combined = 0.5 * cls_probs + 0.5 * q_probs
                pred = int(combined.argmax())
                if pred == agent.reverse_map[task]:
                    ok += 1
            acc[task.value] = ok / rounds
    acc['average'] = sum(acc.values()) / len(acc) if acc else 0
    return acc


def eval_real_samples(agent) -> dict:
    """独立真实样本测试（无偏评估）。"""
    results = {}
    value_to_intent = {it.value: it for it in IntentType}
    with torch.no_grad():
        for label_str, text in agent.EVAL_REAL_SAMPLES.items():
            clean = label_str.rstrip("_0123456789")
            intent = value_to_intent.get(clean)
            if intent is None:
                continue
            stub = AgentState()
            stub.increment_round()
            vec = agent._extract_features(text, stub)
            cls_probs = agent._classify_probs(vec)
            q_vals = agent.q_network(torch.FloatTensor(vec).unsqueeze(0))
            q_probs = torch.softmax(q_vals, dim=-1).squeeze(0).numpy()
            combined = 0.5 * cls_probs + 0.5 * q_probs
            pred_idx = int(combined.argmax())
            pred_intent = agent.action_map[pred_idx]
            correct = pred_idx == agent.reverse_map[intent]
            results[text[:25]] = {
                "predicted": pred_intent.value,
                "expected": intent.value,
                "correct": correct,
            }
    return results


def format_confusion(results: dict) -> str:
    lines = []
    for text, info in results.items():
        mark = "✓" if info["correct"] else "✗"
        lines.append(f"  {mark} [{info['expected']:>12}] → {info['predicted']:<12} | {text}")
    return "\n".join(lines)


def main():
    eval_only = '--eval-only' in sys.argv
    do_distill = '--distill' in sys.argv
    do_csv = '--csv' in sys.argv
    fresh = '--fresh' in sys.argv
    total = int(sys.argv[1]) if len(sys.argv) > 1 and not eval_only else 200000
    log_every = int(sys.argv[2]) if len(sys.argv) > 2 and not eval_only else 5000

    if fresh and os.path.exists(DEFAULT_MODEL_PATH):
        os.remove(DEFAULT_MODEL_PATH)
        print(f"  [fresh] 已删除旧模型")

    agent = AgentRL()

    if eval_only:
        print(f"\n  [评估] 模型 {DEFAULT_MODEL_PATH} (ε={agent.epsilon:.4f}, 步数={agent.training_step})")
        acc = eval_accuracy(agent)
        for k, v in acc.items():
            print(f"    模板 {k}: {v:.2%}")
        real = eval_real_samples(agent)
        print(f"\n  真实样本 ({sum(1 for r in real.values() if r['correct'])}/{len(real)} 正确):")
        print(format_confusion(real))
        return

    if do_csv:
        agent.train_from_csv(epochs=15, augment_factor=3)

    if do_distill:
        agent.distill_from_llm(LLM_DISTILL_LABELS, num_augment=8)
        agent.save()

    mode = "全新训练" if not agent.loaded_from_disk else "续训"
    print(f"\n  [开始] {mode}，目标 {total} 步，ε={agent.epsilon:.4f}")

    t0 = time.time()
    templates_list = list(agent.TASK_TEMPLATES.keys())
    files = agent.SAMPLE_FILES

    for step in range(1, total + 1):
        task = random.choice(templates_list)
        f = random.choice(files)
        g = random.choice([x for x in files if x != f])
        query = random.choice(agent.TASK_TEMPLATES[task]).format(f=f, g=g, c=agent.SAMPLE_CONTENT)

        stub = AgentState()
        stub.increment_round()
        vec = agent._extract_features(query, stub)

        if random.random() < agent.epsilon:
            action = random.randint(0, agent.num_actions - 1)
        else:
            with torch.no_grad():
                q = agent.q_network(torch.FloatTensor(vec).unsqueeze(0))
                action = q.argmax().item()

        correct = agent.reverse_map[task]
        reward = 5.0 if action == correct else -2.0
        done = action == correct
        if done:
            reward += 10.0

        agent.store_experience(vec, action, reward, vec, done)
        agent.train_step()

        if step % log_every == 0:
            acc = eval_accuracy(agent)
            agent.save()
            elapsed = (time.time() - t0) / 60.0
            parts = [f"{k}:{v:.0%}" for k, v in acc.items() if k != 'average']
            print(f"  [步 {step}/{total}] ε={agent.epsilon:.4f} "
                  f"步数={agent.training_step} {elapsed:.1f}m "
                  f"avg={acc['average']:.1%} | {', '.join(parts)}")

    acc = eval_accuracy(agent)
    agent.save()
    print("\n" + "=" * 70)
    print(f"  训练完成 {total} 步，耗时 {(time.time()-t0)/60:.1f}m")
    print("  模板准确率:")
    for k, v in acc.items():
        print(f"    {k}: {v:.2%}")

    real = eval_real_samples(agent)
    print(f"\n  真实样本 ({sum(1 for r in real.values() if r['correct'])}/{len(real)} 正确):")
    print(format_confusion(real))
    print(f"  模型已保存 → {DEFAULT_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
