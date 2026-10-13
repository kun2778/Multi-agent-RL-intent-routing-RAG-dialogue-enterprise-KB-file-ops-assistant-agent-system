"""重复代码门禁 / 重复检测 的单元测试。

覆盖两件事：

1. ``analyze.find_duplicates`` 必须遍历 ``ast.ClassDef`` 内部方法 ——
   旧实现只看模块顶层函数，类方法（``AgentRL.process``、
   ``MetaAgent._semantic_conflict_guard``…）一条都测不出来，
   duplicate_pairs 恒为 0；
2. ``duplicate_gate`` —— 把 review_*/findings.csv 的 duplicate 检测接入 CI：
   新增重复对必须被拦下、基线内的已知债务放行、评审归档要被读进来。
"""

import ast
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import analyze
import duplicate_gate as dg

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))


def _file_data(pairs):
    """[(相对路径, 源码), ...] → find_duplicates 需要的最小输入。"""
    return [{"rel": rel, "tree": ast.parse(src)} for rel, src in pairs]


# ================================================================
# 1. analyze：重复检测必须看得见类方法
# ================================================================
def test_find_duplicates_walks_class_methods():
    a = "class Alpha:\n    def run(self, x):\n        return x + 1\n"
    b = "class Beta:\n    def run(self, x):\n        return x + 1\n"
    dupes = analyze.find_duplicates(_file_data([("a.py", a), ("b.py", b)]))
    assert len(dupes) == 1, f"类方法重复未被发现: {dupes}"
    first = dupes[0]
    got = {(first["file1"], first["func1"]), (first["file2"], first["func2"])}
    assert ("a.py", "Alpha.run") in got and ("b.py", "Beta.run") in got, \
        f"限定名不对（应为 Class.method）: {dupes}"
    print(f"[OK] analyze 识别类方法重复: {dupes[0]['func1']} ~ {dupes[0]['func2']}")


def test_find_duplicates_still_covers_module_functions():
    src = "def helper(x):\n    return [i for i in x if i]\n"
    dupes = analyze.find_duplicates(_file_data([("a.py", src), ("b.py", src)]))
    assert len(dupes) == 1 and dupes[0]["func1"] == "helper", dupes
    print("[OK] analyze 仍覆盖模块级函数")


# 唯一豁免：一行转发包装器。两个 agent 的 get_intent_label 现在都是
# ``return intent_label(intent)``（AST 仍逐字相同），但标签查找逻辑
# 已收敛到 intent_shared.intent_label 单份，包装器本身没有可漂移的状态。
TRIVIAL_WRAPPERS = {
    frozenset({("agent_rl.py", "AgentRL.get_intent_label"),
               ("marl_agent.py", "MetaAgent.get_intent_label")}),
}


def test_no_verbatim_copy_between_the_two_architectures():
    """两套架构之间不得再有逐字复制（同源漂移的根源）。

    这是本项目的核心断言：agent_rl.py 与 marl_agent.py 里任何一对
    「AST 完全相同」的方法都算失败 —— 它们必然只有一份是对的。
    """
    files = []
    for name in ("agent_rl.py", "marl_agent.py", "intent_shared.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        files.append((name, text))

    dupes = analyze.find_duplicates(_file_data(files))
    cross = [d for d in dupes
             if {d["file1"], d["file2"]} == {"agent_rl.py", "marl_agent.py"}
             and frozenset({(d["file1"], d["func1"]),
                            (d["file2"], d["func2"])}) not in TRIVIAL_WRAPPERS]
    assert not cross, \
        "agent_rl 与 marl_agent 之间仍有逐字复制:\n" + \
        "\n".join(f"  {d['func1']} ~ {d['func2']}" for d in cross)
    print(f"[OK] 两架构之间零逐字复制（跨文件重复 {len(cross)} 对）")


# ================================================================
# 2. duplicate_gate：findings.csv 的 duplicate 检测接入 CI
# ================================================================
def test_pair_key_is_order_insensitive():
    assert dg.pair_key("a.py::f", "b.py::g") == dg.pair_key("b.py::g", "a.py::f")
    print("[OK] 重复对键与顺序无关")


def test_new_duplicates_are_flagged():
    current = [{"key": "a.py::f|b.py::f", "left": "a.py::f", "right": "b.py::f",
                "similarity": 1.0, "exact": True}]
    assert dg.new_duplicates(current, set()) , "新增重复未被识别"
    assert not dg.new_duplicates(current, {"a.py::f|b.py::f"}), "基线内重复被误判为新增"
    print("[OK] 新增重复 vs 基线内已知债务")


def test_archived_review_findings_are_ingested():
    """review_*/findings.csv 的 duplicate 行必须被 CI 读到。"""
    archived = dg.archived_duplicate_pairs(ROOT)
    assert archived, "没有读到任何 review_*/findings.csv 的 duplicate 行"
    key = dg.pair_key("agent_rl.py::get_intent_label",
                      "marl_agent.py::get_intent_label")
    assert key in archived, \
        f"评审已归档的逐字复制对不在归档集合里: {key}"
    print(f"[OK] 归档 duplicate {len(archived)} 对（含评审标记的逐字复制）")


def test_gate_fails_on_unbaselined_duplicate(tmp_path):
    """端到端：临时项目里造一对新重复 → 必须退出码 1。"""
    src = "def shared(a, b):\n    total = 0\n    for i in a:\n        total += i * b\n    return total\n"
    (tmp_path / "x.py").write_text(src, encoding="utf-8")
    (tmp_path / "y.py").write_text(src, encoding="utf-8")
    baseline = tmp_path / "duplicate_baseline.json"
    report = tmp_path / "report.csv"

    # 基线缺失 → 配置错误（退出码 2）
    code, _ = dg.run_gate(report_path=report, root=tmp_path,
                          baseline_path=baseline, verbose=False)
    assert code == dg.EXIT_ERROR, f"缺基线应报 2，实为 {code}"

    # 空基线 → 新增重复 → 拦截（退出码 1）
    dg.save_baseline([], baseline)
    code, result = dg.run_gate(report_path=report, root=tmp_path,
                               baseline_path=baseline, verbose=False)
    assert code == dg.EXIT_FAIL, f"新增重复应拦截，实为 {code}"
    assert result["added"], "结果里没有列出新增重复对"
    assert report.exists(), "没有产出 findings.csv 同 schema 的报告"

    # 把该对收进基线 → 放行（退出码 0）
    dg.save_baseline(result["added"], baseline)
    code, _ = dg.run_gate(report_path=report, root=tmp_path,
                          baseline_path=baseline, verbose=False)
    assert code == dg.EXIT_PASS, f"基线内应放行，实为 {code}"
    print("[OK] 门禁: 缺基线=2 / 新增重复=1 / 基线内=0")


def test_gate_passes_on_current_tree(tmp_path):
    """当前仓库必须过门禁（否则 CI 上就是红的）。"""
    code, result = dg.run_gate(report_path=tmp_path / "dup.csv",
                               root=ROOT, verbose=False)
    assert result.get("error") is None, result
    assert code == dg.EXIT_PASS, \
        f"当前代码存在未入基线的重复: {result.get('added')}"
    assert result["total"] > 0 and result["archived_total"] > 0, result
    print(f"[OK] 当前 {result['total']} 对重复全部在基线内，"
          f"评审归档 {result['archived_total']} 对")


def test_gate_scans_only_the_committed_tree(tmp_path):
    """采集口径必须跟 git 一致：.gitignore 掉的脚手架不扫，待入库的照扫。"""
    if shutil.which("git") is None:
        print("[SKIP] PATH 里没有 git，跳过 git 口径采集测试")
        return

    init = subprocess.run(["git", "init", "-q"], cwd=str(tmp_path),
                          capture_output=True)
    assert init.returncode == 0, init.stderr.decode("utf-8", "replace")

    src = ("def build_table(a):\n"
           "    total = 0\n"
           "    for i in a:\n"
           "        total += i * 2\n"
           "    return total\n")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "one.py").write_text(src, encoding="utf-8")
    (tmp_path / "scratch" / "two.py").write_text(src, encoding="utf-8")
    (tmp_path / "main.py").write_text("import os\n", encoding="utf-8")

    # 脚手架进了 .gitignore → 不得采集（否则本地门禁红于 CI）
    (tmp_path / ".gitignore").write_text("scratch/\n", encoding="utf-8")
    rels = [rel for rel, _ in dg.collect_source_files(tmp_path)]
    assert not any(r.startswith("scratch/") for r in rels), rels
    assert "main.py" in rels, rels

    # 取消忽略 → 未跟踪但属于待入库内容，必须被扫到
    (tmp_path / ".gitignore").unlink()
    rels = [rel for rel, _ in dg.collect_source_files(tmp_path)]
    assert {"scratch/one.py", "scratch/two.py"} <= set(rels), rels
    print("[OK] 采集口径 = git 口径（ignore 的不扫 / 待入库的照扫）")


def test_collect_keeps_review_prefixed_source_files(tmp_path):
    """``review_`` 前缀只该排除目录，别误杀 review_collector.py 这类源码。"""
    if shutil.which("git") is None:
        print("[SKIP] PATH 里没有 git，跳过")
        return
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), capture_output=True)
    (tmp_path / "review_pipeline.py").write_text("x = 1\n", encoding="utf-8")
    rels = [rel for rel, _ in dg.collect_source_files(tmp_path)]
    assert "review_pipeline.py" in rels, rels
    print("[OK] review_*.py 源码未被当归档目录排除")


def run_all_tests():
    print("\n" + "#" * 70)
    print("  重复检测 / duplicate_gate 测试")
    print("#" * 70)
    test_find_duplicates_walks_class_methods()
    test_find_duplicates_still_covers_module_functions()
    test_no_verbatim_copy_between_the_two_architectures()
    test_pair_key_is_order_insensitive()
    test_new_duplicates_are_flagged()
    test_archived_review_findings_are_ingested()
    return True


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for name in ("e2e", "current", "gitrepo", "reviewname"):
            (Path(td) / name).mkdir()
        test_gate_fails_on_unbaselined_duplicate(Path(td) / "e2e")
        test_gate_passes_on_current_tree(Path(td) / "current")
        test_gate_scans_only_the_committed_tree(Path(td) / "gitrepo")
        test_collect_keeps_review_prefixed_source_files(Path(td) / "reviewname")
    run_all_tests()
