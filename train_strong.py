"""长期高效训练脚本 v2：蒸馏 + 分类预训练 + DQN 强化。

用法:
  python train_strong.py [steps] [log_every] [--agent single|marl] [--fresh]
                          [--eval-only] [--distill] [--csv]
                          [--ablate-nsp] [--nsp-ablate <档位>]
    steps         single: 训练迭代次数（默认 200000）；marl: 自博弈回合数（默认 20）
    log_every     每多少步打印并保存（默认 5000，仅 single）
    --agent       single（默认，DQN 离线循环）或 marl（走 MetaAgent.train_episode）
    --fresh       删除旧模型重新训练
    --eval-only   仅评估（默认续训）
    --distill     先用蒸馏标签预训练分类头，再 RL 微调
    --csv         用 datasets/intent_corpus.csv 训练分类头（推荐先跑）
    --ablate-nsp  等价于 --nsp-ablate off：整块关掉 NSP
    --nsp-ablate  逐项消融档位，取值见 nsp_latent.ABLATION_SPECS：
                  none / off / no_ema / no_infonce / no_bridge /
                  no_hstep / no_imagine / deterministic
                  对照实验两侧必须用同一 steps/log_every，只改这一个开关。
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
from intent_shared import eval_pool, train_pool, train_reward
from nsp_latent import ABLATION_SPECS, apply_nsp_ablation, nsp_enabled


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
    """模板准确率 —— 只跑留出模板（训练阶段不可见），否则训练集=测试集。"""
    acc = {}
    with torch.no_grad():
        for task, all_templates in agent.TASK_TEMPLATES.items():
            templates = eval_pool(all_templates)
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


def print_nsp(agent) -> None:
    """打印 NSP 潜空间读数：消融对照里「关掉的到底是什么」就靠这两段。"""
    spec = getattr(agent, "nsp_ablate", "none")
    state = "开启" if nsp_enabled(agent) else "关闭（消融）"
    print(f"\n  NSP 潜空间 ({state}，档位={spec})")
    probe = agent.nsp_probe()
    if probe:
        print("    探针（确定性：同一份权重两次调用逐位一致）:")
        for key, value in probe.items():
            print(f"      {key:<22} {value:.6f}")
    snap = agent.nsp_metrics_snapshot()
    if snap:
        print("    训练滚动指标（EMA 均值）:")
        for key, value in snap.items():
            print(f"      {key:<22} {value:.6f}")


def _flag_value(flag: str, default=None):
    """取 ``--flag value`` 或 ``--flag=value``；两个都没给则返回 default。"""
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


# 这些开关吃掉后面那个 token，否则 'none' / 'marl' 会被当成 steps 触发 int() 崩溃
_VALUE_FLAGS = ("--agent", "--nsp-ablate")


def _positional_args() -> list:
    out, skip = [], False
    for tok in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if tok in _VALUE_FLAGS:
            skip = True
            continue
        if tok.startswith("--"):
            continue
        out.append(tok)
    return out


def run_marl(total: int, nsp_spec, eval_only: bool, fresh: bool) -> None:
    """MARL 侧消融：三专家走 ``MetaAgent.train_episode`` 的模板自博弈预热。"""
    from marl_agent import DEFAULT_MARL_MODEL_PATH, create_marl_agent

    if fresh and os.path.exists(DEFAULT_MARL_MODEL_PATH):
        os.remove(DEFAULT_MARL_MODEL_PATH)
        print("  [fresh] 已删除旧 MARL 模型")

    meta = create_marl_agent()
    if nsp_spec:
        changed = apply_nsp_ablation(meta, nsp_spec)
        print(f"  [消融] NSP 档位={nsp_spec}，改动：{', '.join(changed) or '(无)'}")
    elif getattr(meta, "nsp_ablate", "none") != "none":
        print(f"  [提醒] checkpoint 自带 NSP 消融档位={meta.nsp_ablate}")

    report = meta._league_tournament()
    if eval_only:
        print(f"\n  [评估] 模型 {meta.model_path} "
              f"(ε={meta.epsilon:.4f}, 步数={meta.training_step})")
        print(f"    联盟评估 accuracy={report['accuracy']:.2%} "
              f"h2h={report['h2h']:.3f}")
        print_nsp(meta)
        return

    print(f"\n  [架构] MARL 三专家 + NSP (MAPPO, Looped T=4, "
          f"NSP={'on' if nsp_enabled(meta) else 'off(消融)'}, "
          f"档位={meta.nsp_ablate})")
    print(f"  [开始] 联盟式预热 {total} 回合，ε={meta.epsilon:.4f} "
          f"(训练前 accuracy={report['accuracy']:.2%})")

    meta.train_episode(total)
    report = meta._league_tournament()
    print("\n" + "=" * 70)
    print(f"  MARL 训练完成 {total} 回合，"
          f"联盟 accuracy={report['accuracy']:.2%} h2h={report['h2h']:.3f}")
    print_nsp(meta)
    print("=" * 70)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        return

    # 位置参数与开关分开解析：开关若落在第一个位置上，
    # 旧写法 int(sys.argv[1]) 会直接抛 ValueError
    positional = _positional_args()
    eval_only = '--eval-only' in sys.argv
    do_distill = '--distill' in sys.argv
    do_csv = '--csv' in sys.argv
    fresh = '--fresh' in sys.argv

    agent_kind = (_flag_value('--agent', 'single') or 'single').strip().lower()
    nsp_spec = _flag_value('--nsp-ablate', None)
    if nsp_spec is None and '--ablate-nsp' in sys.argv:
        nsp_spec = 'off'
    if nsp_spec:
        nsp_spec = nsp_spec.strip().lower()
        if nsp_spec not in ABLATION_SPECS:
            print(f"  [✗] 未知 --nsp-ablate {nsp_spec}，"
                  f"可选：{', '.join(ABLATION_SPECS)}")
            return
    if agent_kind not in ('single', 'marl'):
        print(f"  [✗] 未知 --agent {agent_kind}，可选：single / marl")
        return

    if agent_kind == 'marl':
        # MARL 没有单步离线循环，positional[0] 解释成自博弈回合数
        total = int(positional[0]) if positional and not eval_only else 20
        run_marl(total, nsp_spec, eval_only, fresh)
        return

    total = int(positional[0]) if positional and not eval_only else 200000
    log_every = int(positional[1]) if len(positional) > 1 and not eval_only else 5000

    if fresh and os.path.exists(DEFAULT_MODEL_PATH):
        os.remove(DEFAULT_MODEL_PATH)
        print(f"  [fresh] 已删除旧模型")

    agent = AgentRL()
    if nsp_spec:
        changed = apply_nsp_ablation(agent, nsp_spec)
        print(f"  [消融] NSP 档位={nsp_spec}，改动：{', '.join(changed) or '(无)'}")
    elif agent.nsp_ablate != "none":
        print(f"  [提醒] checkpoint 自带 NSP 消融档位={agent.nsp_ablate}，"
              f"要跑对照请显式传 --nsp-ablate none")

    if eval_only:
        print(f"\n  [评估] 模型 {DEFAULT_MODEL_PATH} (ε={agent.epsilon:.4f}, 步数={agent.training_step})")
        acc = eval_accuracy(agent)
        for k, v in acc.items():
            print(f"    模板 {k}: {v:.2%}")
        real = eval_real_samples(agent)
        print(f"\n  真实样本 ({sum(1 for r in real.values() if r['correct'])}/{len(real)} 正确):")
        print(format_confusion(real))
        print_nsp(agent)
        return

    if do_csv:
        agent.train_from_csv(epochs=15, augment_factor=3)

    if do_distill:
        agent.distill_from_llm(LLM_DISTILL_LABELS, num_augment=8)
        agent.save()

    mode = "全新训练" if not agent.loaded_from_disk else "续训"
    print(f"\n  [开始] {mode}，目标 {total} 步，ε={agent.epsilon:.4f}")
    print(f"  [架构] Transformer Q-network + NSP (Looped T=6, MoE E=8, "
          f"λ_nsp={agent.nsp_lambda}, NSP={'on' if nsp_enabled(agent) else 'off(消融)'}"
          f"档位={agent.nsp_ablate}, Engram)")

    t0 = time.time()
    templates_list = list(agent.TASK_TEMPLATES.keys())
    files = agent.SAMPLE_FILES

    for step in range(1, total + 1):
        task = random.choice(templates_list)
        f = random.choice(files)
        g = random.choice([x for x in files if x != f])
        # 只在训练池采样（评估走 eval_accuracy 的留出池）
        query = random.choice(train_pool(agent.TASK_TEMPLATES[task])).format(
            f=f, g=g, c=agent.SAMPLE_CONTENT)

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
        is_correct = action == correct
        reward = train_reward(is_correct)
        # 单步自博弈：一步即一个 episode，next_state 与 state 相同，
        # done=False 会退化成「用同一个状态 bootstrap 自己」的 TD。
        done = True

        agent.store_experience(vec, action, reward, vec, done)
        agent.train_step()

        # Engram 写入: 只在答对（正例）时积累经验
        if is_correct and hasattr(agent, 'engram'):
            with torch.no_grad():
                feat_tensor = torch.FloatTensor(vec).unsqueeze(0)
                cls_hidden = agent.q_network.transformer(feat_tensor)[:, 0]
                conf_tensor = torch.FloatTensor([[1.0]])
                agent.engram.write(cls_hidden, feat_tensor, conf_tensor)

        if step % log_every == 0:
            acc = eval_accuracy(agent)
            agent.save()
            elapsed = (time.time() - t0) / 60.0
            parts = [f"{k}:{v:.0%}" for k, v in acc.items() if k != 'average']
            nsp_snap = agent.nsp_metrics_snapshot()
            nsp_part = (f" | nsp_total={nsp_snap.get('nsp_total', float('nan')):.3f}"
                        f" nsp_acc={nsp_snap.get('nsp_infonce_acc', float('nan')):.3f}"
                        if nsp_snap else "")
            print(f"  [步 {step}/{total}] ε={agent.epsilon:.4f} "
                  f"步数={agent.training_step} {elapsed:.1f}m "
                  f"avg={acc['average']:.1%}{nsp_part} | {', '.join(parts)}")

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
    print_nsp(agent)
    print(f"  模型已保存 → {DEFAULT_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
