"""重复代码门禁 —— 把 review_*/findings.csv 的 duplicate 检测接入 CI。

背景
----
离线评审产物 ``review_*/findings.csv`` 里有一批 ``category=duplicate`` 的条目
（相似度 ≥ 0.82 的函数/方法对），但它们只躺在人工报告里：没人会回头读，
CI 也不看。于是「agent_rl 与 marl_agent 犯相同的错误」第一形态——逐字复制
（特征提取器、冲突护栏、路由表、模板库、投票聚合……11+ 处）可以在两套
架构里各留一份、越修越多，而项目自带的评审其实**早就标记过**。

本脚本把同一套检测搬到每次提交上跑，并用基线（白名单）区分
「已知债务」与「新增重复」：只有**新增**才拦截。

组成
----
- **检测器** ``relation_analyzer.find_duplicate_functions``
  —— 与 findings.csv 的 duplicate 行**同源**（同一函数、同一阈值 0.82）；
  结果按 sim == 1.0 再分成「逐字复制」与「近似重复」两档。
- **已归档清单** ``review_*/findings.csv`` 中 ``category=duplicate`` 的
  evidence 行 —— 用于标注每一对是「评审已知」还是「新冒出来的」，
  并统计已消除的存量债务。
- **基线** ``eval/regression/duplicate_baseline.json`` —— 受版本控制，
  CI checkout 即可拿到（.runtime/ 被 gitignore，不能放这里）。

判定
----
- 当前重复对 ∉ 基线 → **新增重复** → 退出码 1 → CI 拦截合入；
- 基线里已消失的重复对 → 报告为「已消除」，提示 ``--update-baseline`` 收编；
- 基线缺失 → 退出码 2（与 regression_gate 一致：缺基线属于配置错误）。

用法::

    python duplicate_gate.py                    # 跑门禁（CI 默认行为）
    python duplicate_gate.py --update-baseline  # 固化当前重复对（人工触发）
    python duplicate_gate.py --json             # 机器可读输出

退出码：0 通过 / 1 有新增重复 / 2 出错（缺基线等）
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from atomic_io import atomic_write_json, atomic_write_text
from content_parser import parse_file
from relation_analyzer import SimilarityHit, find_duplicate_functions
from runtime_paths import PROJECT_ROOT, RUNTIME_ROOT

# ---------------------------------------------------------------- 路径
GATE_DIR = PROJECT_ROOT / "eval" / "regression"
BASELINE_PATH = GATE_DIR / "duplicate_baseline.json"
DEFAULT_REPORT_PATH = RUNTIME_ROOT / "duplicate_findings.csv"

#: 与 relation_analyzer / review 报告一致的相似度阈值
SIMILARITY_THRESHOLD = 0.82

#: findings.csv 的列顺序（report_writer.write_reports 的 finding_fields）
FINDING_FIELDS = [
    "id", "priority", "severity", "category", "file", "line", "language",
    "directory", "archive", "score", "confidence", "description",
    "evidence", "recommendation", "conditions", "risk_reasons", "git_authors",
]

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

# 与 analyze.py 同口径的排除目录，外加历史评审快照目录。
# 主口径是下面 _git_py_files 的「git 视角」（.gitignore 里的一律不扫）；
# 这里再兜一层：非 git 检出（zip 下载、git 不可用）走目录遍历时同样不会
# 扫到本地脚手架 .gitignore 只是没写进这些名字而已。
EXCLUDE_DIRS = {".git", "__pycache__", ".runtime", ".venv", "node_modules",
                ".pytest_cache", ".trash", ".backup", ".idea", ".vscode",
                "AgentRL_BranchTest"}


# ================================================================ 检测
def _git_py_files(root: Path) -> Optional[List[str]]:
    """git 视角的 .py 清单：已跟踪 + 未跟踪但未被 .gitignore 忽略。

    与 CI 检出（git clone）同口径 —— 本地临时脚手架只要进了 .gitignore
    就不会被门禁扫到，新写的、准备入库的文件仍会被扫到。

    root 必须本身就是仓库顶层（否则 git 输出的相对路径以顶层为基准，
    会读错文件、甚至静默漏检）；非 git 检出 / git 不可用时返回 None，
    由调用方回退到目录遍历。
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, timeout=30)
        if top.returncode != 0:
            return None
        toplevel = top.stdout.decode("utf-8", "replace").strip()
        if os.path.normcase(os.path.normpath(toplevel)) != \
                os.path.normcase(os.path.normpath(str(root))):
            return None
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z",
             "--cached", "--others", "--exclude-standard", "--", "*.py"],
            capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]


def _excluded(rel: str) -> bool:
    """路径（正斜杠）是否落在排除名单 / review_* 归档里。

    ``review_`` 前缀只对**目录**生效（与目录遍历口径一致），否则
    ``review_collector.py`` / ``review_pipeline.py`` 这两个正经源码会被误杀。
    """
    *dirs, fname = rel.split("/")
    return any(d in EXCLUDE_DIRS or d.startswith("review_") for d in dirs) \
        or fname in EXCLUDE_DIRS


def collect_source_files(root: Path = PROJECT_ROOT) -> List[Tuple[str, str]]:
    """采集待检源码，返回 [(相对路径, 文件内容), ...]（只收 .py）。

    优先用 git 清单（``_git_py_files``），保证「入库口径」：.gitignore 掉的
    临时目录不扫、待入库的未跟踪文件照扫；非 git 检出时回退到目录遍历。

    排除 ``review_*`` 评审快照：里面是历史归档（还带着
    ``refactored_files.zip!…`` 之类的压缩包成员路径），拿它当被检对象
    会把「已经搬走的旧副本」重新算成当前重复。
    """
    out: List[Tuple[str, str]] = []
    git_paths = _git_py_files(root)
    if git_paths is not None:
        candidates = sorted(p for p in git_paths if not _excluded(p))
    else:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in EXCLUDE_DIRS and not d.startswith("review_")
            )
            for fname in sorted(filenames):
                if fname.endswith(".py"):
                    candidates.append(
                        (Path(dirpath) / fname).relative_to(root).as_posix())
        candidates.sort()
    for rel in candidates:
        path = Path(root) / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.append((rel, text))
    return out


def pair_key(left: str, right: str) -> str:
    """归一化的重复对键：两侧按字典序排序，避免 (a,b)/(b,a) 各算一对。"""
    return "|".join(sorted((left, right)))


def _side(file_path: str, func: str) -> str:
    return f"{file_path.replace(chr(92), '/')}::{func}"


def detect_duplicates(root: Path = PROJECT_ROOT,
                      threshold: float = SIMILARITY_THRESHOLD
                      ) -> List[Dict[str, Any]]:
    """跑 findings.csv 同款检测，返回按相似度降序的重复对列表。

    每项含 ``key``（归一化键）、``left``/``right``、``similarity``、
    ``exact``（是否逐字/结构完全相同）。
    """
    sources = collect_source_files(root)
    contents: Dict[str, str] = {}
    parsed = []
    for rel, text in sources:
        contents[rel] = text
        parsed.append(parse_file(rel, text))

    hits: List[SimilarityHit] = find_duplicate_functions(
        parsed, contents, threshold=threshold)

    results: List[Dict[str, Any]] = []
    for h in hits:
        left = _side(h.file1, h.func1)
        right = _side(h.file2, h.func2)
        results.append({
            "key": pair_key(left, right),
            "left": left,
            "right": right,
            "similarity": float(h.similarity),
            "exact": float(h.similarity) >= 1.0,
        })
    return results


# ================================================================ 已归档评审
def archived_duplicate_pairs(root: Path = PROJECT_ROOT
                             ) -> Dict[str, Dict[str, Any]]:
    """读取 review_*/findings.csv 的 category=duplicate 行。

    返回 ``{归一化键: {"evidence": ..., "source": findings.csv 路径}}``。
    压缩包成员（``xxx.zip!path.py::f``）原样保留：它们只用于「已知/已消除」
    统计，不参与当前源码的比对。
    """
    pairs: Dict[str, Dict[str, Any]] = {}
    for review_dir in sorted(root.glob("review_*")):
        csv_path = review_dir / "findings.csv"
        if not csv_path.is_file():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("category") or "").strip() != "duplicate":
                        continue
                    evidence = (row.get("evidence") or "").strip()
                    key = _key_from_evidence(evidence)
                    if key:
                        pairs.setdefault(
                            key, {"evidence": evidence,
                                  "source": csv_path.relative_to(root).as_posix()})
        except (OSError, csv.Error):
            continue
    return pairs


_EVIDENCE_RE = None


def _key_from_evidence(evidence: str) -> str:
    """``a.py::f ~ b.py::g sim=0.91`` → 归一化键。"""
    global _EVIDENCE_RE
    if _EVIDENCE_RE is None:
        import re
        _EVIDENCE_RE = re.compile(
            r"^(?P<a>.+?)\s*~\s*(?P<b>.+?)\s+sim=\d+(?:\.\d+)?\s*$")
    if not evidence:
        return ""
    m = _EVIDENCE_RE.match(evidence)
    if not m:
        return ""
    return pair_key(m.group("a").strip(), m.group("b").strip())


# ================================================================ 基线
def load_baseline(path: Path = BASELINE_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "pairs" not in data:
        raise ValueError(f"基线格式不合法: {path}")
    return data


def save_baseline(pairs: Sequence[str], path: Path = BASELINE_PATH,
                  threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Any]:
    data = {
        "threshold": threshold,
        "note": ("已知重复对白名单：由 duplicate_gate.py --update-baseline 生成。"
                 "新增重复对不在其中 → CI 拦截；修复存量后重跑即可收窄。"),
        "pairs": sorted(set(pairs)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)
    return data


# ================================================================ 报告
def build_report_rows(current: List[Dict[str, Any]],
                      archived: Dict[str, Dict[str, Any]],
                      baseline: set) -> List[Dict[str, str]]:
    """生成 findings.csv 同 schema 的重复明细（供 CI 作为 artifact 上传）。"""
    rows: List[Dict[str, str]] = []
    for i, item in enumerate(sorted(current, key=lambda x: -x["similarity"]), 1):
        documented = item["key"] in archived
        known = item["key"] in baseline
        rows.append({
            "id": f"D{i:03d}",
            "priority": "1" if item["exact"] else "3",
            "severity": "HIGH" if item["exact"] else "MEDIUM",
            "category": "duplicate",
            "file": item["left"].split("::", 1)[0],
            "line": "",
            "language": "python",
            "directory": ".",
            "archive": "",
            "score": int(item["similarity"] * 60),
            "confidence": "0.9",
            "description": (f"与 {item['right']} 相似度 "
                            f"{item['similarity']}（阈值 {SIMILARITY_THRESHOLD}）"),
            "evidence": f"{item['left']} ~ {item['right']} sim={item['similarity']}",
            "recommendation": "抽取公共模块消除重复",
            "conditions": f"similarity>={SIMILARITY_THRESHOLD}",
            "risk_reasons": "|".join(
                ["重复代码", "逐字复制" if item["exact"] else "近似重复",
                 "基线内" if known else "新增",
                 "评审已归档" if documented else "未见于评审报告"]),
            "git_authors": "",
        })
    return rows


def write_report_csv(rows: List[Dict[str, str]],
                     path: Path = DEFAULT_REPORT_PATH) -> Path:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FINDING_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in FINDING_FIELDS})
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, buf.getvalue(), encoding="utf-8-sig")
    return path


# ================================================================ 门禁
def new_duplicates(current: List[Dict[str, Any]], baseline_pairs: set
                   ) -> List[Dict[str, Any]]:
    return [c for c in current if c["key"] not in baseline_pairs]


def run_gate(update_baseline: bool = False,
             report_path: Path = DEFAULT_REPORT_PATH,
             root: Path = PROJECT_ROOT,
             baseline_path: Path = BASELINE_PATH,
             verbose: bool = True) -> Tuple[int, Dict[str, Any]]:
    """执行门禁，返回 ``(退出码, 结果字典)``。"""
    current = detect_duplicates(root)
    archived = archived_duplicate_pairs(root)

    if update_baseline:
        save_baseline([c["key"] for c in current], baseline_path)
        if verbose:
            print(f"  [重复门禁] 已更新基线 {baseline_path.name}："
                  f"{len(current)} 对")
        baseline_pairs = {c["key"] for c in current}
    else:
        if not baseline_path.exists():
            print(f"  [重复门禁] 缺少基线: {baseline_path}\n"
                  f"            先跑 python duplicate_gate.py --update-baseline")
            return EXIT_ERROR, {"error": "missing_baseline"}
        baseline_pairs = set(load_baseline(baseline_path)["pairs"])

    added = new_duplicates(current, baseline_pairs)
    resolved = sorted(baseline_pairs - {c["key"] for c in current})
    still_archived = sorted(set(archived) & {c["key"] for c in current})
    archived_gone = sorted(set(archived) - {c["key"] for c in current})

    rows = build_report_rows(current, archived, baseline_pairs)
    report_file = write_report_csv(rows, report_path)

    exact = sum(1 for c in current if c["exact"])
    result = {
        "total": len(current),
        "exact": exact,
        "near": len(current) - exact,
        "baseline": len(baseline_pairs),
        "added": [a["key"] for a in sorted(added, key=lambda x: -x["similarity"])],
        "resolved_in_baseline": resolved,
        "archived_total": len(archived),
        "archived_still_present": len(still_archived),
        "archived_resolved": len(archived_gone),
        "report": str(report_file),
    }

    if verbose:
        print(f"  [重复门禁] 当前 {len(current)} 对（逐字 {exact} / "
              f"近似 {len(current) - exact}），基线 {len(baseline_pairs)} 对")
        print(f"  [重复门禁] review_*/findings.csv 归档 {len(archived)} 对："
              f"仍存在 {len(still_archived)}、已消除 {len(archived_gone)}")
        print(f"  [重复门禁] 报告: {report_file}")
        if resolved:
            print(f"  [重复门禁] 基线中已消除 {len(resolved)} 对，"
                  f"可 --update-baseline 收窄基线")
        if added:
            print(f"  [重复门禁] 新增重复 {len(added)} 对（不在基线内）:")
            for a in sorted(added, key=lambda x: -x["similarity"])[:20]:
                tag = "逐字" if a["exact"] else "近似"
                doc = "（评审已归档）" if a["key"] in archived else ""
                print(f"      [{tag}] {a['left']} ~ {a['right']} "
                      f"sim={a['similarity']}{doc}")
            print("  [重复门禁] 抽取到共享模块后重跑；"
                  "确属可接受的重复才用 --update-baseline 收编。")

    if added:
        return EXIT_FAIL, result
    return EXIT_PASS, result


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description="重复代码门禁（findings.csv duplicate 检测接入 CI）")
    parser.add_argument("--update-baseline", action="store_true",
                        help="把当前重复对固化进基线（人工触发）")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH),
                        help="重复明细 CSV 输出路径（findings.csv 同 schema）")
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    args = parser.parse_args(argv)

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    code, result = run_gate(
        update_baseline=args.update_baseline,
        report_path=Path(args.report),
        # --json 时关掉人类可读日志：stdout 必须是纯 JSON，
        # 否则 CI 用 ``| tee duplicate-gate.json`` 会拿到一个解析不了的文件。
        verbose=not args.json,
    )
    if args.json:
        print(json.dumps({"exit_code": code, **result},
                         ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
