"""输出组件：唯一目录、MD/JSON/CSV、摘要、索引、只读摘要文件、校验。"""

import csv
import hashlib
import io
import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from decision_engine import DecisionReport, Finding
from relation_analyzer import RelationReport, ComplexityHit, SimilarityHit, CrossRefHit
from atomic_io import atomic_write_text


def _json_default(o: Any):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, bytes):
        return o.decode('utf-8', errors='replace')
    return str(o)


def make_review_dir(base_dir: str, fingerprint_src: str = "") -> str:
    """创建唯一目录 review_YYYYMMDD_HHmmss_<short_hash>。绝不覆盖已有目录。"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    h = hashlib.sha1(
        f"{fingerprint_src}|{ts}|{os.getpid()}".encode('utf-8')
    ).hexdigest()[:8]
    base = os.path.join(base_dir, f"review_{ts}_{h}")
    path = base
    n = 1
    while os.path.exists(path):
        path = f"{base}_{n}"
        n += 1
    os.makedirs(path, exist_ok=False)
    # 只读标记（目录内文件仍可写入报告，源项目不动）
    return path


def write_json(path: str, data: Any) -> Dict[str, Any]:
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    atomic_write_text(path, payload, encoding='utf-8')
    return {'path': path, 'bytes': len(payload.encode('utf-8')), 'sha1': hashlib.sha1(payload.encode()).hexdigest()}


def write_csv(path: str, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> Dict[str, Any]:
    if not rows:
        rows = []
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ['id']
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in fieldnames})
    text = buf.getvalue()
    atomic_write_text(path, text, encoding='utf-8-sig')
    return {'path': path, 'bytes': len(text.encode('utf-8')), 'rows': len(rows)}


def build_markdown(
    project: str,
    generated: str,
    collection_summary: Dict,
    git_summary: str,
    decision: DecisionReport,
    relations: RelationReport,
    archives: List[Dict],
    skipped: List[Dict],
    runtime_summary: Dict,
) -> str:
    s = decision.summary
    lines: List[str] = []
    lines.append(f"# {project} 代码审查报告")
    lines.append('')
    lines.append(f"- 生成时间: {generated}")
    lines.append(f"- 模式: 只读（不修改/不移动/不删除/不重命名/不覆盖）")
    lines.append(f"- 扫描文件: {collection_summary.get('files', 0)} "
                 f"(可读 {collection_summary.get('readable', 0)}, "
                 f"跳过 {collection_summary.get('skipped', 0)})")
    lines.append(f"- 子项目: {', '.join(collection_summary.get('subprojects', [])[:20]) or '无'}")
    lines.append(f"- 忽略源: {', '.join(collection_summary.get('ignore_sources', [])) or '无'}")
    lines.append(f"- 发现: HIGH {s.get('high', 0)} / MEDIUM {s.get('medium', 0)} / LOW {s.get('low', 0)}")
    lines.append('')

    lines.append('## 执行摘要')
    lines.append(
        f"本次只读审查覆盖 {s.get('files_analyzed', 0)} 个可解析文件，"
        f"产出 {s.get('total_findings', 0)} 条发现（高 {s.get('high', 0)}、"
        f"中 {s.get('medium', 0)}、低 {s.get('low', 0)}）。"
        f"识别循环依赖 {len(relations.circular_deps)} 组、"
        f"高复杂度函数 {len(relations.complex_functions)} 个、"
        f"相似度≥0.82 重复对 {len(relations.duplicate_pairs)}、"
        f"孤儿模块 {len(relations.orphan_modules)} 个。"
        f"条件交集命中 {len(decision.condition_intersections)} 组。"
    )
    lines.append('')

    lines.append('## 采集范围')
    lines.append('| 项目 | 值 |')
    lines.append('|------|----|')
    for k, v in collection_summary.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)[:200]
        lines.append(f'| {k} | {v} |')
    lines.append('')

    lines.append('## Git 历史')
    lines.append('```')
    lines.append(git_summary)
    lines.append('```')
    lines.append('')

    lines.append('## 风险分级')
    for level in ('HIGH', 'MEDIUM', 'LOW'):
        fs = [f for f in decision.findings if f.severity == level]
        if not fs:
            continue
        lines.append(f'### {level}（{len(fs)}）')
        for f in fs[:40]:
            loc = f"{f.file}:{f.line}" if f.line else f.file
            lines.append(f"{f.priority}. [{f.category}] {loc} — {f.description}")
        if len(fs) > 40:
            lines.append(f"... 另有 {len(fs) - 40} 条")
        lines.append('')

    lines.append('## 条件交集')
    if decision.condition_intersections:
        for name, paths in decision.condition_intersections.items():
            lines.append(f"- **{name}**: {', '.join(paths[:15])}"
                         + (f" (+{len(paths)-15})" if len(paths) > 15 else ''))
    else:
        lines.append('- 无')
    lines.append('')

    lines.append('## 分组')
    lines.append('### 按语言')
    for lang, ids in sorted(decision.groups_by_language.items(), key=lambda x: -len(x[1])):
        lines.append(f"- {lang}: {len(ids)} 条")
    lines.append('### 按目录（Top 15）')
    dirs = sorted(decision.groups_by_directory.items(), key=lambda x: -len(x[1]))[:15]
    for d, ids in dirs:
        lines.append(f"- `{d}`: {len(ids)} 条")
    lines.append('### 按压缩包归属')
    for a, ids in sorted(decision.groups_by_archive.items(), key=lambda x: -len(x[1])):
        lines.append(f"- {a}: {len(ids)} 条")
    lines.append('### 按风险等级')
    for lv in ('HIGH', 'MEDIUM', 'LOW'):
        lines.append(f"- {lv}: {len(decision.groups_by_risk.get(lv, []))} 条")
    lines.append('')

    lines.append('## 依赖拓扑（修复顺序建议）')
    order = decision.topological_order[:40]
    if order:
        for i, p in enumerate(order, 1):
            lines.append(f"{i}. {p}")
        if len(decision.topological_order) > 40:
            lines.append(f"... 共 {len(decision.topological_order)} 个节点")
    else:
        lines.append('- 无依赖边')
    lines.append('')

    lines.append('## 循环依赖')
    if relations.circular_deps:
        for c in relations.circular_deps:
            lines.append('- ' + ' → '.join(c))
    else:
        lines.append('- 无')
    lines.append('')

    lines.append('## 高复杂度函数（≥15）')
    if relations.complex_functions:
        lines.append('| 文件 | 函数 | 行 | 复杂度 |')
        lines.append('|------|------|----|--------|')
        for h in relations.complex_functions[:30]:
            lines.append(f'| {h.file} | {h.name} | {h.line} | {h.complexity} |')
    else:
        lines.append('- 无')
    lines.append('')

    lines.append('## 重复代码（相似度 ≥ 0.82）')
    if relations.duplicate_pairs:
        for d in relations.duplicate_pairs[:20]:
            lines.append(f"- {d.file1}::{d.func1} ↔ {d.file2}::{d.func2} ({d.similarity})")
    else:
        lines.append('- 无')
    lines.append('')

    lines.append('## 压缩包')
    if archives:
        lines.append('| 路径 | 有效 | 格式 | 内部文件 | 跳过 |')
        lines.append('|------|------|------|----------|------|')
        for a in archives:
            lines.append(
                f"| {a.get('path')} | {a.get('valid')} | {a.get('format', '')} "
                f"| {a.get('inner_files', 0)} | {a.get('skipped', 0)} |"
            )
    else:
        lines.append('- 未发现压缩包或已跳过')
    lines.append('')

    lines.append('## 跳过记录（含原因）')
    if skipped:
        lines.append('| 路径 | 类别 | 原因 |')
        lines.append('|------|------|------|')
        # 聚合类别计数 + 前 50 条明细
        from collections import Counter
        cats = Counter(x.get('category', '') for x in skipped)
        lines.append(f"| （汇总） {len(skipped)} 条 | {dict(cats)} | |")
        for sk in skipped[:50]:
            lines.append(f"| {sk.get('path')} | {sk.get('category')} | {sk.get('reason')} |")
        if len(skipped) > 50:
            lines.append(f"| ... 另有 {len(skipped) - 50} 条 | | |")
    else:
        lines.append('- 无跳过记录')
    lines.append('')

    lines.append('## 运行时状态')
    lines.append('```json')
    lines.append(json.dumps(runtime_summary, ensure_ascii=False, indent=2, default=_json_default))
    lines.append('```')
    lines.append('')

    lines.append('## 产物清单')
    lines.append('- report.md — 本报告')
    lines.append('- index.json — 结构化索引与元数据')
    lines.append('- findings.csv — 发现明细')
    lines.append('- files.csv — 文件清单')
    lines.append('- archive_index.csv — 压缩包内文件索引')
    lines.append('- skips.csv — 跳过记录')
    lines.append('- dependency_graph.json — 依赖图与拓扑')
    lines.append('- summaries/*.md — 按原相对路径命名的只读摘要')
    lines.append('')
    return '\n'.join(lines)


def write_file_summaries(out_dir: str, parsed_list: List, contents: Dict[str, str],
                         max_files: int = 500) -> List[str]:
    """按原相对路径结构写只读摘要（summaries/ 下保持相对路径）。"""
    written = []
    sum_root = os.path.join(out_dir, 'summaries')
    count = 0
    for pf in parsed_list:
        if count >= max_files:
            break
        rel = pf.path.replace('/', os.sep).replace('!', '__')
        # 防路径穿越
        rel = rel.replace('..', '_')
        dest = os.path.join(sum_root, rel + '.md')
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            text = contents.get(pf.path, '')
            body = [
                f"# 摘要: {pf.path}",
                '',
                f"- 语言: {pf.language}",
                f"- 大小: {pf.size} 字节",
                f"- 行数: {pf.line_count}",
                f"- 编码: {pf.encoding}",
                f"- 导入: {', '.join(pf.imports[:20]) or '无'}",
                f"- 公开 API: {', '.join(pf.public_apis[:20]) or '无'}",
                f"- TODO: {len(pf.todos)}",
                f"- 环境变量: {', '.join(pf.env_refs[:20]) or '无'}",
                f"- URL: {len(pf.urls)}",
                f"- 密钥疑似: {len(pf.suspected_secrets)}",
                '',
                '## 内容预览（只读）',
                '```',
                text[:2000],
                '```' if len(text) <= 2000 else '```\n...（已截断）',
            ]
            atomic_write_text(dest, '\n'.join(body), encoding='utf-8')
            written.append(os.path.relpath(dest, out_dir).replace('\\', '/'))
            count += 1
        except OSError:
            continue
    return written


def write_reports(
    out_dir: str,
    project: str,
    collection_summary: Dict,
    git_summary: str,
    decision: DecisionReport,
    relations: RelationReport,
    files_rows: List[Dict],
    archives: List[Dict],
    archive_index: List[Dict],
    skipped: List[Dict],
    parsed_list: List,
    contents: Dict[str, str],
    runtime_summary: Dict,
) -> Dict[str, Any]:
    """写出全部报告产物并返回清单 + 校验信息。"""
    generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    artifacts: Dict[str, Any] = {}

    # findings.csv
    finding_fields = [
        'id', 'priority', 'severity', 'category', 'file', 'line', 'language',
        'directory', 'archive', 'score', 'confidence', 'description',
        'evidence', 'recommendation', 'conditions', 'risk_reasons', 'git_authors',
    ]
    finding_rows = []
    for f in decision.findings:
        finding_rows.append({
            'id': f.id, 'priority': f.priority, 'severity': f.severity,
            'category': f.category, 'file': f.file, 'line': f.line,
            'language': f.language, 'directory': f.directory, 'archive': f.archive,
            'score': f.score, 'confidence': f.confidence,
            'description': f.description, 'evidence': f.evidence,
            'recommendation': f.recommendation,
            'conditions': '|'.join(f.conditions),
            'risk_reasons': '|'.join(f.risk_reasons),
            'git_authors': '|'.join(f.git_authors),
        })
    artifacts['findings_csv'] = write_csv(
        os.path.join(out_dir, 'findings.csv'), finding_rows, finding_fields)

    # files.csv
    file_fields = ['path', 'size', 'mtime', 'encoding', 'readable', 'in_archive',
                   'language_ext', 'subproject']
    artifacts['files_csv'] = write_csv(os.path.join(out_dir, 'files.csv'),
                                       files_rows, file_fields)

    # archive_index.csv
    arc_fields = ['archive', 'inner_path', 'virtual_path', 'size', 'mtime', 'format']
    artifacts['archive_index_csv'] = write_csv(
        os.path.join(out_dir, 'archive_index.csv'), archive_index, arc_fields)

    # skips.csv
    skip_fields = ['path', 'category', 'reason']
    artifacts['skips_csv'] = write_csv(os.path.join(out_dir, 'skips.csv'),
                                       skipped, skip_fields)

    # dependency_graph.json
    artifacts['dep_graph_json'] = write_json(os.path.join(out_dir, 'dependency_graph.json'), {
        'graph': relations.dependency_graph,
        'reverse': relations.reverse_graph,
        'cycles': relations.circular_deps,
        'orphans': relations.orphan_modules,
        'topological_order': relations.topological_order,
        'unused_exports': relations.unused_exports[:200],
    })

    # index.json
    index = {
        'meta': {
            'project': project,
            'generated': generated,
            'mode': 'readonly',
            'short_hash': os.path.basename(out_dir).split('_')[-1],
            'review_dir': os.path.basename(out_dir),
        },
        'collection': collection_summary,
        'git_summary': git_summary,
        'summary': decision.summary,
        'condition_intersections': decision.condition_intersections,
        'groups': {
            'by_language': decision.groups_by_language,
            'by_directory': decision.groups_by_directory,
            'by_archive': decision.groups_by_archive,
            'by_risk': decision.groups_by_risk,
        },
        'topological_order': decision.topological_order,
        'archives': archives,
        'findings': [
            {
                'id': f.id, 'priority': f.priority, 'severity': f.severity,
                'category': f.category, 'file': f.file, 'line': f.line,
                'description': f.description, 'recommendation': f.recommendation,
                'score': f.score, 'confidence': f.confidence,
                'conditions': f.conditions, 'risk_reasons': f.risk_reasons,
                'dependencies': f.dependencies,
            }
            for f in decision.findings
        ],
        'skipped_count': len(skipped),
        'skipped_sample': skipped[:100],
        'runtime': runtime_summary,
        'artifacts': {},
    }
    artifacts['index_json'] = write_json(os.path.join(out_dir, 'index.json'), index)

    # report.md
    md = build_markdown(
        project=project, generated=generated,
        collection_summary=collection_summary, git_summary=git_summary,
        decision=decision, relations=relations, archives=archives,
        skipped=skipped, runtime_summary=runtime_summary,
    )
    md_path = os.path.join(out_dir, 'report.md')
    atomic_write_text(md_path, md, encoding='utf-8')
    artifacts['report_md'] = {
        'path': md_path,
        'bytes': len(md.encode('utf-8')),
        'sha1': hashlib.sha1(md.encode('utf-8')).hexdigest(),
    }

    # 只读摘要
    summary_files = write_file_summaries(out_dir, parsed_list, contents)
    artifacts['summaries'] = summary_files

    # 回填 index artifacts
    index['artifacts'] = {
        k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'path'})
        for k, v in artifacts.items() if k != 'summaries'
    }
    index['artifacts']['summaries_count'] = len(summary_files)
    write_json(os.path.join(out_dir, 'index.json'), index)

    return artifacts


def validate_reports(out_dir: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """报告生成后校验：文件存在、JSON 可解析、CSV 行数一致。"""
    checks = []
    ok = True

    def check(name: str, cond: bool, detail: str = ''):
        nonlocal ok
        checks.append({'name': name, 'ok': bool(cond), 'detail': detail})
        if not cond:
            ok = False

    report_md = os.path.join(out_dir, 'report.md')
    index_json = os.path.join(out_dir, 'index.json')
    findings_csv = os.path.join(out_dir, 'findings.csv')

    check('report.md exists', os.path.isfile(report_md),
          f"{os.path.getsize(report_md) if os.path.isfile(report_md) else 0} bytes")
    check('index.json exists', os.path.isfile(index_json))
    check('findings.csv exists', os.path.isfile(findings_csv))

    findings_count = -1
    if os.path.isfile(index_json):
        try:
            with open(index_json, 'r', encoding='utf-8') as f:
                idx = json.load(f)
            findings_count = len(idx.get('findings', []))
            check('index.json parses', True)
            check('index.meta present', 'meta' in idx)
        except (json.JSONDecodeError, OSError) as e:
            check('index.json parses', False, str(e))

    if os.path.isfile(findings_csv):
        try:
            with open(findings_csv, 'r', encoding='utf-8-sig') as f:
                rows = list(csv.reader(f))
            csv_count = max(len(rows) - 1, 0)
            check('findings.csv row match',
                  findings_count < 0 or csv_count == findings_count,
                  f"csv={csv_count} index={findings_count}")
        except (OSError, csv.Error) as e:
            check('findings.csv readable', False, str(e))

    if os.path.isfile(report_md):
        with open(report_md, 'r', encoding='utf-8') as f:
            md = f.read()
        check('report.md non-trivial', len(md) > 500, f"{len(md)} chars")
        check('report.md has risk section', '风险分级' in md)
        check('report.md has skip section', '跳过记录' in md)

    # 不覆盖校验：目录应可列举
    try:
        listing = os.listdir(out_dir)
        check('output dir listable', len(listing) >= 4, f"{len(listing)} entries")
    except OSError as e:
        check('output dir listable', False, str(e))

    return {'ok': ok, 'checks': checks}
