"""关系与度量分析：依赖图、相似度、复杂度、循环依赖、孤儿、AST 级交叉引用。"""

import ast
import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from content_parser import ParsedFile, detect_language


@dataclass
class ComplexityHit:
    file: str
    name: str
    line: int
    complexity: int


@dataclass
class SimilarityHit:
    file1: str
    file2: str
    func1: str
    func2: str
    similarity: float


@dataclass
class CrossRefHit:
    kind: str  # api_test | config_env | url_doc | secret_risk
    left: str
    right: str
    detail: str = ''


@dataclass
class RelationReport:
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    reverse_graph: Dict[str, List[str]] = field(default_factory=dict)
    circular_deps: List[List[str]] = field(default_factory=list)
    orphan_modules: List[str] = field(default_factory=list)
    unused_exports: List[str] = field(default_factory=list)
    complex_functions: List[ComplexityHit] = field(default_factory=list)
    duplicate_pairs: List[SimilarityHit] = field(default_factory=list)
    unhandled_exceptions: List[Dict] = field(default_factory=list)
    todo_priority: List[Dict] = field(default_factory=list)
    cross_references: List[CrossRefHit] = field(default_factory=list)
    topological_order: List[str] = field(default_factory=list)


def _module_keys(rel_path: str) -> List[str]:
    """文件可被 import 的候选模块名。"""
    p = rel_path.replace('\\', '/').rstrip('/')
    keys = set()
    if p.endswith('.py'):
        stem = p[:-3]
        parts = stem.split('/')
        if parts[-1] == '__init__':
            parts = parts[:-1]
        keys.add(parts[-1])
        keys.add('.'.join(parts))
        if len(parts) > 1:
            keys.add(parts[-1])
    else:
        base = os.path.basename(p)
        keys.add(os.path.splitext(base)[0])
        # 包相对引用 provider.base 等
        noext = p
        for ext in ('.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs'):
            if noext.endswith(ext):
                noext = noext[: -len(ext)]
                break
        parts = noext.split('/')
        keys.add(parts[-1])
        keys.add('.'.join(parts))
    return [k for k in keys if k]


def build_dependency_graph(parsed: List[ParsedFile]) -> Dict[str, List[str]]:
    """构建项目内依赖图（import 解析到项目文件）。"""
    key_to_files: Dict[str, List[str]] = defaultdict(list)
    all_paths = {pf.path for pf in parsed}
    for pf in parsed:
        for k in _module_keys(pf.path):
            key_to_files[k].append(pf.path)

    graph: Dict[str, List[str]] = {}
    for pf in parsed:
        deps: Set[str] = set()
        for imp in pf.imports:
            # 相对导入
            base_dir = os.path.dirname(pf.path)
            candidates = []
            if imp.startswith('.'):
                # python relative: .mod ..pkg.mod
                dots = len(imp) - len(imp.lstrip('.'))
                mod = imp.lstrip('.')
                parent = base_dir
                for _ in range(max(dots - 1, 0)):
                    parent = os.path.dirname(parent) or ''
                if mod:
                    candidates.append(f"{parent}/{mod.replace('.', '/')}.py".replace('\\', '/').lstrip('/'))
                    candidates.append(f"{parent}/{mod.split('.')[0]}/__init__.py".replace('\\', '/'))
                else:
                    candidates.append(f"{parent}/__init__.py".replace('\\', '/'))
            else:
                top = imp.split('/')[0].split('.')[0]
                candidates.append(f"{top}.py")
                candidates.append(f"{top}/__init__.py")
                # 全限定
                cand = imp.replace('.', '/') + '.py'
                candidates.append(cand)
                if key_to_files.get(imp):
                    for f in key_to_files[imp]:
                        if f != pf.path:
                            deps.add(f)
                if key_to_files.get(top):
                    for f in key_to_files[top]:
                        if f != pf.path:
                            deps.add(f)
            for c in candidates:
                c = c.replace('\\', '/').lstrip('./')
                # 尝试相对 base_dir
                joined = os.path.normpath(os.path.join(base_dir, c)).replace('\\', '/') if not c.startswith('/') else c
                for cand in (c, joined, os.path.normpath(c).replace('\\', '/')):
                    if cand in all_paths and cand != pf.path:
                        deps.add(cand)
        if deps:
            graph[pf.path] = sorted(deps)
    return graph


def build_reverse_graph(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    rev: Dict[str, List[str]] = defaultdict(list)
    for src, dsts in graph.items():
        for d in dsts:
            rev[d].append(src)
    return {k: sorted(v) for k, v in rev.items()}


def find_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = defaultdict(int)
    cycles: List[List[str]] = []
    path: List[str] = []

    def dfs(node: str):
        color[node] = GRAY
        path.append(node)
        for nb in graph.get(node, []):
            if color[nb] == GRAY:
                idx = path.index(nb)
                cycles.append(path[idx:] + [nb])
            elif color[nb] == WHITE:
                dfs(nb)
        path.pop()
        color[node] = BLACK

    for n in list(graph.keys()):
        if color[n] == WHITE:
            dfs(n)

    seen = set()
    unique = []
    for c in cycles:
        key = tuple(sorted(set(c[:-1])))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def topological_sort(graph: Dict[str, List[str]], all_nodes: List[str]) -> List[str]:
    """Kahn 拓扑排序；环内节点追加在末尾。"""
    nodes = set(all_nodes) | set(graph.keys())
    for deps in graph.values():
        nodes.update(deps)
    indeg = {n: 0 for n in nodes}
    rev: Dict[str, List[str]] = defaultdict(list)
    for src, dsts in graph.items():
        for d in dsts:
            # edge src -> d means src depends on d; 修复顺序应先处理依赖
            indeg[src] = indeg.get(src, 0)  # ensure
            rev[d].append(src)
            indeg[d] = indeg.get(d, 0)
    # 重新计算：indeg[x] = 依赖的出边数（x 依赖多少）
    indeg = {n: 0 for n in nodes}
    dependents: Dict[str, List[str]] = defaultdict(list)
    for src, dsts in graph.items():
        indeg[src] = len(dsts)
        for d in dsts:
            dependents[d].append(src)
    queue = sorted([n for n in nodes if indeg.get(n, 0) == 0])
    order: List[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for dep in dependents.get(n, []):
            indeg[dep] -= 1
            if indeg[dep] == 0:
                queue.append(dep)
        queue.sort()
    remaining = sorted(nodes - set(order))
    order.extend(remaining)
    return order


def find_orphans(graph: Dict[str, List[str]], all_paths: List[str],
                 rev: Dict[str, List[str]], protected: Optional[Set[str]] = None) -> List[str]:
    protected = protected or set()
    orphans = []
    for p in all_paths:
        if p.endswith('__init__.py'):
            continue
        if p in protected:
            continue
        if not rev.get(p):
            orphans.append(p)
    return sorted(orphans)


def find_unused_exports(parsed: List[ParsedFile], graph: Dict[str, List[str]]) -> List[str]:
    """公开 API 未被其他文件引用。"""
    imported_tokens: Set[str] = set()
    for pf in parsed:
        imported_tokens.update(pf.imports)
        # from x import name 模式已在 imports 中只有模块，补充扫描无法拿到
    unused = []
    for pf in parsed:
        for api in pf.public_apis:
            short = api.split('.')[-1]
            # 在其他文件的 import 行无法精确匹配；用模块名粗匹配
            if short not in imported_tokens and pf.path not in {d for deps in graph.values() for d in deps}:
                # 被依赖文件中的 API 仍未使用
                pass
            if short not in imported_tokens:
                # 检查是否有任何其他文件依赖此文件
                dependents = [s for s, deps in graph.items() if pf.path in deps]
                if not dependents:
                    unused.append(f"{pf.path}::{api}")
    return unused


def cyclomatic_complexity(tree: ast.AST, threshold: int = 15) -> List[ComplexityHit]:
    """函数圈复杂度（不进入嵌套函数体重复计算）。"""
    hits: List[ComplexityHit] = []

    def score(node: ast.AST) -> int:
        c = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.IfExp)):
                c += 1
            elif isinstance(child, ast.BoolOp):
                c += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                c += 1
            elif isinstance(child, ast.Match):
                c += max(len(child.cases) - 1, 0)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child is not node:
                # 不深入嵌套定义
                continue
        return c

    def walk_funcs(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 仅统计直接子节点中的分支 + 自身体（排除嵌套函数）
                c = 1
                for sub in ast.walk(child):
                    if sub is child:
                        continue
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        # 跳过嵌套定义的内容
                        continue
                    if isinstance(sub, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.IfExp)):
                        c += 1
                    elif isinstance(sub, ast.BoolOp):
                        c += len(sub.values) - 1
                    elif isinstance(sub, ast.comprehension):
                        c += 1
                if c >= threshold:
                    hits.append(ComplexityHit(file='', name=child.name, line=child.lineno, complexity=c))
                walk_funcs(child)
            elif isinstance(child, ast.ClassDef):
                walk_funcs(child)
            elif isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                walk_funcs(child)

    walk_funcs(tree)
    return hits


def _norm_tokens(src: str) -> Set[str]:
    return set(re.findall(r'[A-Za-z_][A-Za-z0-9_]+|\d+', src))


def _func_source_and_nodes(node: ast.AST) -> Tuple[str, str]:
    dump = re.sub(r'line=\d+', 'line=0', ast.dump(node))
    dump = re.sub(r'col_offset=\d+', 'col_offset=0', dump)
    return dump, dump


def find_duplicate_functions(
    parsed: List[ParsedFile],
    contents: Dict[str, str],
    threshold: float = 0.82,
) -> List[SimilarityHit]:
    """跨文件函数相似度 >= threshold。"""
    funcs: List[Tuple[str, str, ast.AST]] = []
    for pf in parsed:
        if pf.language != 'python':
            continue
        src = contents.get(pf.path)
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append((pf.path, node.name, node))

    # 分桶减少比较：按 token 大小区间
    entries = []
    for path, name, node in funcs:
        dump = re.sub(r'line=\d+|col_offset=\d+', '0', ast.dump(node))
        tokens = _norm_tokens(dump)
        entries.append((path, name, tokens, dump))

    hits: List[SimilarityHit] = []
    seen_pairs: Set[Tuple] = set()
    n = len(entries)
    for i in range(n):
        p1, n1, t1, d1 = entries[i]
        if len(t1) < 5:
            continue
        for j in range(i + 1, n):
            p2, n2, t2, d2 = entries[j]
            if p1 == p2 and n1 == n2:
                continue
            if len(t2) < 5:
                continue
            if d1 == d2:
                sim = 1.0
            else:
                inter = len(t1 & t2)
                union = len(t1 | t2)
                sim = inter / union if union else 0.0
            if sim >= threshold:
                key = tuple(sorted([f"{p1}::{n1}", f"{p2}::{n2}"]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                hits.append(SimilarityHit(p1, p2, n1, n2, round(sim, 3)))
    hits.sort(key=lambda h: -h.similarity)
    return hits


def find_unhandled_exceptions(parsed: List[ParsedFile], contents: Dict[str, str]) -> List[Dict]:
    """裸 except 且未 raise/return 处理的可疑点。"""
    results: List[Dict] = []
    for pf in parsed:
        if pf.language != 'python':
            continue
        src = contents.get(pf.path)
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                bare = node.type is None
                handles = False
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Raise, ast.Return, ast.Continue, ast.Pass)):
                        handles = True
                        break
                # Pass 视为弱处理
                weak = bare and any(isinstance(s, ast.Pass) for s in ast.walk(node))
                if bare and not handles:
                    results.append({
                        'file': pf.path, 'line': node.lineno,
                        'detail': '裸 except 未处理',
                    })
                elif bare and weak:
                    results.append({
                        'file': pf.path, 'line': node.lineno,
                        'detail': '裸 except 仅 pass',
                    })
    return results


def build_todo_priority(parsed: List[ParsedFile]) -> List[Dict]:
    """TODO 优先级排序：FIXME>HACK>XXX>TODO，文件风险加权。"""
    weight = {'FIXME': 40, 'HACK': 30, 'XXX': 20, 'TODO': 10}
    items = []
    for pf in parsed:
        for t in pf.todos:
            items.append({
                'file': pf.path,
                'line': t['line'],
                'marker': t['marker'],
                'text': t['text'],
                'score': weight.get(t['marker'], 5),
            })
    items.sort(key=lambda x: -x['score'])
    for i, it in enumerate(items, 1):
        it['priority'] = i
    return items


def cross_reference(parsed: List[ParsedFile], contents: Dict[str, str]) -> List[CrossRefHit]:
    """AST/内容级交叉分析。"""
    hits: List[CrossRefHit] = []
    apis_by_file = {pf.path: pf.public_apis for pf in parsed}
    test_files = [pf for pf in parsed if 'test' in pf.path.lower() or pf.path.startswith('test')]
    env_by_file = {pf.path: pf.env_refs for pf in parsed}
    config_by_file = {pf.path: (pf.config_keys, pf.language) for pf in parsed}
    url_by_file = {pf.path: pf.urls for pf in parsed}
    secret_by_file = {pf.path: pf.suspected_secrets for pf in parsed}
    doc_files = [pf for pf in parsed if pf.language in ('markdown', 'rst') or pf.path.endswith('.md')]

    # 1) 公开 API ↔ 测试引用
    for tf in test_files:
        src = contents.get(tf.path, '')
        for fpath, apis in apis_by_file.items():
            if fpath == tf.path:
                continue
            if 'test' in fpath.lower():
                continue
            for api in apis:
                short = api.split('.')[-1]
                if len(short) < 3:
                    continue
                if re.search(rf'\b{re.escape(short)}\b', src):
                    hits.append(CrossRefHit(
                        kind='api_test',
                        left=f"{fpath}::{api}",
                        right=tf.path,
                        detail='测试引用公开 API',
                    ))

    # 2) 配置键 ↔ 环境变量（同文件或同目录）
    for fpath, (keys, lang) in config_by_file.items():
        if not keys:
            continue
        envs = env_by_file.get(fpath, [])
        # 相邻 env 文件
        dirn = os.path.dirname(fpath)
        for ef, evs in env_by_file.items():
            if os.path.dirname(ef) == dirn or ef == fpath:
                envs = list(set(envs) | set(evs))
        if keys and envs:
            for k in keys[:20]:
                ku = k.upper().replace('.', '_').replace('-', '_')
                for e in envs:
                    if e and (e in ku or ku.startswith(e) or e.replace('_', '') in k.replace('.', '').replace('-', '')):
                        hits.append(CrossRefHit(
                            kind='config_env',
                            left=f"{fpath}::{k}",
                            right=e,
                            detail='配置键与环境变量关联',
                        ))
                        break

    # 3) 硬编码 URL ↔ 文档引用
    all_doc_text = '\n'.join(contents.get(df.path, '') for df in doc_files)
    for fpath, urls in url_by_file.items():
        if fpath in {df.path for df in doc_files}:
            continue
        for u in urls:
            # 文档中出现相同 host/path 片段
            frag = u.rstrip('/')
            if frag and frag in all_doc_text:
                hits.append(CrossRefHit(
                    kind='url_doc',
                    left=f"{fpath}",
                    right=u,
                    detail='URL 同时出现在文档中',
                ))

    # 4) 密钥疑似项 ↔ 风险等级（在 decision 层提升；此处产出交叉项）
    for fpath, secrets in secret_by_file.items():
        for s in secrets:
            hits.append(CrossRefHit(
                kind='secret_risk',
                left=f"{fpath}:{s['line']}",
                right='HIGH',
                detail=s['text'][:120],
            ))

    return hits


def analyze_relations(
    parsed: List[ParsedFile],
    contents: Dict[str, str],
    similarity_threshold: float = 0.82,
    complexity_threshold: int = 15,
) -> RelationReport:
    rep = RelationReport()
    graph = build_dependency_graph(parsed)
    rep.dependency_graph = graph
    rep.reverse_graph = build_reverse_graph(graph)
    rep.circular_deps = find_cycles(graph)
    all_paths = [pf.path for pf in parsed]
    py_paths = [p for p in all_paths if p.endswith('.py')]
    rep.orphan_modules = find_orphans(graph, py_paths, rep.reverse_graph)
    rep.topological_order = topological_sort(graph, all_paths)

    for pf in parsed:
        if pf.language != 'python':
            continue
        src = contents.get(pf.path)
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for hit in cyclomatic_complexity(tree, threshold=complexity_threshold):
            hit.file = pf.path
            rep.complex_functions.append(hit)
    rep.complex_functions.sort(key=lambda h: -h.complexity)

    rep.duplicate_pairs = find_duplicate_functions(parsed, contents, threshold=similarity_threshold)
    rep.unhandled_exceptions = find_unhandled_exceptions(parsed, contents)
    rep.todo_priority = build_todo_priority(parsed)
    rep.cross_references = cross_reference(parsed, contents)
    rep.unused_exports = find_unused_exports(parsed, graph)
    return rep
