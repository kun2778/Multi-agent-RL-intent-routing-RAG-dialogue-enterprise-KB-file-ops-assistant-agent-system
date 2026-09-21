"""Test script for MARL (Multi-Agent Reinforcement Learning) system.

This script tests the MARL agent on various queries to verify it correctly
classifies intents, including the problematic query that failed before.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_state import AgentState, IntentType, QuerySubIntent
from marl_agent import MetaAgent, create_marl_agent
from agent_rl import AgentRL

# Fix encoding for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')


def test_single_query(query: str, expected_intent: str = None):
    """Test a single query with both architectures."""
    print(f"\n{'='*70}")
    print(f"  测试查询: {query}")
    print(f"{'='*70}")

    # Test MARL Agent
    print("\n[MARL 多Agent架构]")
    marl_agent = create_marl_agent()
    state = AgentState()
    state.increment_round()

    intent, path, confidence, clarifications = marl_agent.process(query, state)
    print(f"  意图: {intent.value}")
    print(f"  路径: {path.value if path else 'N/A'}")
    print(f"  置信度: {confidence:.4f}")
    if clarifications:
        print(f"  澄清: {clarifications}")

    # Test Single Agent
    print("\n[单Agent架构]")
    single_agent = AgentRL()
    state2 = AgentState()
    state2.increment_round()

    intent2, path2, confidence2, clarifications2 = single_agent.process(query, state2)
    print(f"  意图: {intent2.value}")
    print(f"  路径: {path2.value if path2 else 'N/A'}")
    print(f"  置信度: {confidence2:.4f}")

    # Compare
    if intent == intent2:
        print(f"\n[结果] OK 两种架构预测一致: {intent.value}")
    else:
        print(f"\n[结果] DIFF 预测不一致: MARL={intent.value}, Single={intent2.value}")

    # Check against expected
    if expected_intent:
        if intent.value == expected_intent:
            print(f"  [OK] MARL 预测正确 (期望: {expected_intent})")
        else:
            print(f"  [FAIL] MARL 预测错误 (期望: {expected_intent}, 实际: {intent.value})")

    return intent.value


def run_all_tests():
    """Run comprehensive tests."""
    print("\n" + "#"*70)
    print("  MARL 系统测试")
    print("#"*70)

    # Test cases: (query, expected_intent)
    test_cases = [
        # Original problematic query
        ("帮我把最近的截屏内容的文字进行识别并输出", "query"),

        # OCR-related queries
        ("识别截屏中的文字", "query"),
        ("提取图片内容", "query"),

        # File operations
        ("添加文件 notes.txt 内容为 你好", "add_file"),
        ("删除文件 old.txt", "delete_file"),
        ("修改文件 config.yaml", "modify_file"),
        ("整理文件夹", "refactor"),

        # Query operations
        ("搜索强化学习", "query"),
        ("查看这个文件", "query"),

        # Deep search
        ("对比分析 a.txt 和 b.txt", "deep_search"),

        # Vague
        ("你好", "vague"),
    ]

    correct = 0
    total = len(test_cases)

    for query, expected in test_cases:
        result = test_single_query(query, expected)
        if result == expected:
            correct += 1

    print("\n" + "="*70)
    print(f"  测试结果: {correct}/{total} 正确 ({correct/total*100:.1f}%)")
    print("="*70)

    return correct == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
