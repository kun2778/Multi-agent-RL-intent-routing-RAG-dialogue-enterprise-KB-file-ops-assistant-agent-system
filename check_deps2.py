import sys
sys.stdout.reconfigure(encoding='utf-8')

packages = {
    'numpy': 'numpy',
    'torch': 'torch',
    'faiss': 'faiss',
    'openai': 'openai',
}

print("\n=== 依赖检查 ===")
all_ok = True
for name, pkg in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  {name}: {ver} [OK]")
    except ImportError:
        print(f"  {name}: NOT INSTALLED [MISSING]（faiss 缺失时自动使用 numpy 回退）")
        if name != 'faiss':
            all_ok = False

try:
    from agent_rl import AgentRL
    agent = AgentRL()
    print(f"\n  agent_rl: 导入成功 [OK]")
    print(f"  QNetwork 输入维度: {agent.q_network.network[0].in_features}")
    print(f"  动作空间: {agent.num_actions} 类")
    print(f"  奖励表: {len(agent.REWARD_MAP)} 种事件")
except Exception as e:
    print(f"  agent_rl: 导入失败 - {e}")
    all_ok = False

try:
    from file_corpus import FileCorpus
    corpus = FileCorpus()
    print(f"\n  file_corpus: 语料库 - {corpus.stats()} [OK]")
except Exception as e:
    print(f"  file_corpus: 导入失败 - {e}")
    all_ok = False

try:
    from fact_query import FactQueryHandler
    handler = FactQueryHandler()
    print(f"  fact_query: FAISS 索引 - {handler.get_stats()} [OK]")
except Exception as e:
    print(f"  fact_query: 导入失败 - {e}")
    all_ok = False

try:
    from deep_search import DeepSearchHandler
    ds = DeepSearchHandler()
    print(f"  deep_search: 语料库 - {ds.get_corpus_stats()} [OK]")
except Exception as e:
    print(f"  deep_search: 导入失败 - {e}")
    all_ok = False

try:
    from answer_generator import AnswerGenerator
    gen = AnswerGenerator()
    print(f"  answer_generator: LLM可用 = {gen.llm_generator.client is not None} [OK]")
except Exception as e:
    print(f"  answer_generator: 导入失败 - {e}")
    all_ok = False

print(f"\n=== 总体: {'全部通过' if all_ok else '有缺失'} ===")
print("建议命令: python main.py （交互对话框）| python main.py \"添加文件 notes.txt 内容为 你好\"")