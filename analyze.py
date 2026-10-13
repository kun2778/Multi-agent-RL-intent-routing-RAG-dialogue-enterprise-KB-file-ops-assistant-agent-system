#!/usr/bin/env python3
"""Static analysis script for AgentRL_IntentRouting project."""
import os, re, ast, json, hashlib, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path(r"C:\Users\84950\Desktop\AgentRL_IntentRouting")
# AgentRL_BranchTest/ 已被 .gitignore 忽略（本地联调脚手架，CI 检出时不存在）
EXCLUDE_DIRS = {".git", "__pycache__", ".runtime", ".venv", "node_modules",
                "AgentRL_BranchTest"}

# ─── helpers ───

def collect_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f

def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")

def last_modified(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

def find_todos(lines):
    """检测未被注释掉的活跃 TODO/FIXME/XXX/HACK。"""
    results = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        m = re.search(r'(TODO|FIXME|XXX|HACK)', stripped, re.IGNORECASE)
        if not m:
            continue
        # 死代码：连续重复注释符
        dead = bool(re.match(r'^(#+\s*){2,}|(//\s*){2,}', stripped))
        if dead:
            continue
        in_comment = stripped.startswith(('#', '//', '--', ';'))
        results.append({
            "line": i, "text": stripped,
            "in_comment": in_comment, "active": True,
            "marker": m.group(1).upper(),
        })
    return results

def calc_complexity(tree):
    """Return list of (func_name, line, complexity)，不重复计入嵌套函数。"""
    results = []

    def walk_funcs(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                c = 1
                for sub in ast.walk(child):
                    if sub is child:
                        continue
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    if isinstance(sub, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.IfExp)):
                        c += 1
                    elif isinstance(sub, ast.BoolOp):
                        c += len(sub.values) - 1
                    elif isinstance(sub, ast.comprehension):
                        c += 1
                if c >= 15:
                    results.append({
                        "name": child.name,
                        "line": child.lineno,
                        "complexity": c,
                    })
                walk_funcs(child)
            elif isinstance(child, ast.ClassDef):
                walk_funcs(child)
            elif isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                walk_funcs(child)

    walk_funcs(tree)
    return results

def extract_imports(tree):
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module.split(".")[0])
    return sorted(set(mods))

def extract_public_apis(tree):
    apis = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            apis.append(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            apis.append(node.name)
    return apis

def find_hardcoded_urls(content):
    return re.findall(r'https?://[^\s\'")\]>]+', content)

def find_env_refs(content):
    refs = re.findall(r'os\.environ\[[\"\']([\w_]+)[\"\']\]', content)
    refs += re.findall(r'os\.environ\.get\([\"\']([\w_]+)[\"\']', content)
    return sorted(set(refs))

def find_secrets(content, lines):
    patterns = re.compile(
        r'(?:api[_-]?key|secret|token|password|passwd|credential|auth[_-]?token)\s*[=:]\s*["\'][^"\']+["\']',
        re.IGNORECASE
    )
    results = []
    for i, line in enumerate(lines, 1):
        if patterns.search(line):
            results.append(line.strip())
    return results

def func_body_hash(node):
    """Return a stable hash of a function body (ignoring names/line numbers)."""
    try:
        code = ast.dump(node)
        # Normalize whitespace & line numbers
        code = re.sub(r'line=\d+', 'line=0', code)
        code = re.sub(r'col_offset=\d+', 'col_offset=0', code)
        return hashlib.md5(code.encode()).hexdigest()
    except:
        return None

def iter_scoped_functions(tree):
    """遍历模块作用域内的所有函数与**类方法**，返回 [(限定名, 节点), ...]。

    旧实现只取 ``ast.iter_child_nodes(tree)`` —— 即模块顶层函数，类方法
    根本进不了候选集。而 agent_rl / marl_agent 的逐字复制（
    ``_safe_fallback``、``_semantic_conflict_guard``、``process``、
    ``_auto_load``…）几乎全是类方法，于是 duplicate_pairs 恒为 0，
    「项目自带评审已标记 11 对」在本脚本里一条都复现不出来。

    限定名 ``Class.method`` 与 findings.csv 的 evidence 对齐，
    便于跨工具比对。
    """
    out = []

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((prefix + child.name, child))
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            elif isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                # 模块级 if __name__ == "__main__": / TYPE_CHECKING 块
                walk(child, prefix)

    walk(tree)
    return out


def find_duplicates(file_data):
    """Compare all function/method pairs across files, return duplicates."""
    # Collect all (file, func_name, ast_node) tuples
    all_funcs = []
    for fd in file_data:
        tree = fd["tree"]
        if tree is None:
            continue
        for name, node in iter_scoped_functions(tree):
            all_funcs.append((fd["rel"], name, node))

    # Hash each function body
    hashes = defaultdict(list)
    for fname, func_name, node in all_funcs:
        h = func_body_hash(node)
        if h:
            hashes[h].append((fname, func_name, node))

    dupes = []
    seen = set()
    for h, group in hashes.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                f1, n1, nd1 = group[i]
                f2, n2, nd2 = group[j]
                key = tuple(sorted([f1, f2, n1, n2]))
                if key in seen:
                    continue
                seen.add(key)
                # 同一哈希分组内：结构完全一致
                set1 = set(re.findall(r'\w+', ast.dump(nd1)))
                set2 = set(re.findall(r'\w+', ast.dump(nd2)))
                if set1 and set2:
                    sim = len(set1 & set2) / max(len(set1 | set2), 1)
                else:
                    sim = 0.0
                sim = max(sim, 1.0)
                if sim >= 0.82:
                    dupes.append({
                        "file1": f1, "file2": f2,
                        "func1": n1, "func2": n2,
                        "similarity": round(sim, 3)
                    })
    return dupes

# ─── Main ───

def build_file_data(root: Path = ROOT):
    files_data = []
    for py in sorted(collect_py_files(ROOT)):
        relpath = rel(py)
        try:
            content = py.read_text(encoding="utf-8", errors="replace")
        except:
            continue
        lines = content.splitlines()
        tree = None
        try:
            tree = ast.parse(content, filename=relpath)
        except SyntaxError:
            pass

        fd = {
            "rel": relpath,
            "tree": tree,
            "path": py,
            "lines": len(lines),
            "todos": find_todos(lines),
            "complex_functions": calc_complexity(tree) if tree else [],
            "imports": extract_imports(tree) if tree else [],
            "public_apis": extract_public_apis(tree) if tree else [],
            "hardcoded_urls": find_hardcoded_urls(content),
            "env_refs": find_env_refs(content),
            "suspected_secrets": find_secrets(content, lines),
            "last_modified": last_modified(py),
        }
        files_data.append(fd)

    return files_data


def main() -> dict:
    """扫描项目、生成 analysis_result.json（导入本模块不会执行）。"""
    files_data = build_file_data()

    # Build output
    output_files = []
    for fd in files_data:
        output_files.append({
            "path": fd["rel"],
            "lines": fd["lines"],
            "todos": fd["todos"],
            "complex_functions": fd["complex_functions"],
            "imports": fd["imports"],
            "public_apis": fd["public_apis"],
            "hardcoded_urls": fd["hardcoded_urls"],
            "env_refs": fd["env_refs"],
            "suspected_secrets": fd["suspected_secrets"],
            "last_modified": fd["last_modified"],
        })

    # ─── Dependency graph ───
    # Map module names to file paths (top-level modules only)
    module_to_files = defaultdict(list)
    for fd in files_data:
        # Get the module stem (filename without .py)
        stem = fd["rel"].replace("/", ".").replace(".py", "")
        # Top-level module: first component
        parts = fd["rel"].split("/")
        if parts[-1] == "__init__.py":
            top = parts[-2] if len(parts) > 1 else "root"
        else:
            top = parts[-1].replace(".py", "")
        module_to_files[top].append(fd["rel"])

    dep_graph = {}
    all_file_rels = {fd["rel"] for fd in files_data}
    for fd in files_data:
        deps = set()
        for imp in fd["imports"]:
            # Try to resolve to a file in the project
            # Check direct module match
            if imp in module_to_files:
                for f in module_to_files[imp]:
                    if f != fd["rel"]:
                        deps.add(f)
            # Check if it's a submodule of a package
            for f in all_file_rels:
                f_stem = f.replace("/", ".").replace(".py", "")
                if f_stem.endswith("." + imp) or f_stem == imp:
                    if f != fd["rel"]:
                        deps.add(f)
        if deps:
            dep_graph[fd["rel"]] = sorted(deps)

    # ─── Circular dependencies ───
    def find_cycles(graph):
        cycles = []
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in graph}
        path = []
        path_set = set()

        def dfs(node):
            color[node] = GRAY
            path.append(node)
            path_set.add(node)
            for nb in graph.get(node, []):
                if nb not in color:
                    color[nb] = WHITE
                if color[nb] == GRAY:
                    # Found cycle
                    idx = path.index(nb)
                    cycle = path[idx:] + [nb]
                    cycles.append(cycle)
                elif color[nb] == WHITE:
                    dfs(nb)
            path.pop()
            path_set.discard(node)
            color[node] = BLACK

        for node in list(graph.keys()):
            if color.get(node, WHITE) == WHITE:
                dfs(node)
        return cycles

    circular = find_cycles(dep_graph)

    # Deduplicate circular deps
    seen_cycles = set()
    unique_circular = []
    for c in circular:
        key = tuple(sorted(c[:-1]))  # exclude duplicate end node
        if key not in seen_cycles:
            seen_cycles.add(key)
            unique_circular.append(c)

    # ─── Orphan modules ───
    all_imported = set()
    for deps in dep_graph.values():
        all_imported.update(deps)

    # A module is orphan if it has a .py file but no other file imports it
    orphan_modules = sorted(
        f for f in all_file_rels
        if f not in all_imported and not f.endswith("__init__.py")
    )

    # ─── Duplicate functions ───
    dupes = find_duplicates(files_data)

    # ─── Final JSON ───
    result = {
        "files": output_files,
        "dependency_graph": dep_graph,
        "circular_deps": unique_circular,
        "orphan_modules": orphan_modules,
        "duplicate_pairs": dupes,
    }

    out_path = ROOT / "analysis_result.json"
    from atomic_io import atomic_write_json
    atomic_write_json(out_path, result)
    print(f"Analysis complete. Output: {out_path}")
    print(f"Files analyzed: {len(output_files)}")
    print(f"Dependencies: {len(dep_graph)} edges")
    print(f"Circular deps: {len(unique_circular)}")
    print(f"Orphan modules: {len(orphan_modules)}")
    print(f"Duplicate pairs: {len(dupes)}")
    return result



if __name__ == "__main__":
    main()
