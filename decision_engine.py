"""决策与分组层：条件交集、风险等级、优先级、拓扑排序、多维分组。"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from content_parser import ParsedFile
from git_history import GitHistoryResult
from relation_analyzer import RelationReport


@dataclass
class Finding:
    id: str = ''
    severity: str = 'LOW'          # HIGH / MEDIUM / LOW
    category: str = ''
    file: str = ''
    line: Optional[int] = None
    description: str = ''
    evidence: str = ''
    recommendation: str = ''
    confidence: float = 0.8
    score: int = 0                 # 越大越优先
    priority: int = 0              # 1 最高
    language: str = ''
    directory: str = ''
    archive: str = ''
    risk_reasons: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    git_authors: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)


@dataclass
class DecisionReport:
    findings: List[Finding] = field(default_factory=list)
    condition_intersections: Dict[str, List[str]] = field(default_factory=dict)
    groups_by_language: Dict[str, List[str]] = field(default_factory=dict)
    groups_by_directory: Dict[str, List[str]] = field(default_factory=dict)
    groups_by_archive: Dict[str, List[str]] = field(default_factory=dict)
    groups_by_risk: Dict[str, List[str]] = field(default_factory=dict)
    topological_order: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def _dir_of(path: str) -> str:
    p = path.replace('\\', '/')
    if '/' not in p:
        return '.'
    return p.rsplit('/', 1)[0]


def _archive_of(path: str) -> str:
    if '!' in path:
        return path.split('!', 1)[0]
    return ''


def _severity_from_score(score: int) -> str:
    if score >= 80:
        return 'HIGH'
    if score >= 40:
        return 'MEDIUM'
    return 'LOW'


def build_findings(
    parsed: List[ParsedFile],
    relations: RelationReport,
    git: Optional[GitHistoryResult] = None,
    archive_members: Optional[Set[str]] = None,
) -> List[Finding]:
    archive_members = archive_members or set()
    findings: List[Finding] = []
    seq = 0
    lang_by_path = {pf.path: pf.language for pf in parsed}

    def add(**kwargs):
        nonlocal seq
        seq += 1
        f = Finding(id=f"F{seq:03d}")
        for k, v in kwargs.items():
            setattr(f, k, v)
        f.language = f.language or lang_by_path.get(f.file, '')
        f.directory = _dir_of(f.file)
        f.archive = _archive_of(f.file) or (f.file if f.file in archive_members else '')
        f.severity = _severity_from_score(f.score)
        findings.append(f)
        return f

    # 高复杂度函数
    for h in relations.complex_functions:
        score = 40 + min(h.complexity, 100)
        add(
            category='complexity',
            file=h.file,
            line=h.line,
            description=f"函数 {h.name} 圈复杂度 {h.complexity}（阈值 15）",
            evidence=f"complexity={h.complexity}",
            recommendation='拆分为多个小函数，降低分支复杂度',
            score=score,
            confidence=0.95,
            risk_reasons=['高复杂度'],
            conditions=['complexity>15'],
        )

    # 循环依赖
    for cyc in relations.circular_deps:
        primary = cyc[0] if cyc else ''
        add(
            category='circular_dependency',
            file=primary,
            description='循环依赖: ' + ' -> '.join(cyc),
            evidence=' -> '.join(cyc),
            recommendation='引入接口层或依赖倒置打破环',
            score=75,
            confidence=0.9,
            risk_reasons=['循环依赖'],
            conditions=['has_cycle'],
            dependencies=cyc,
        )

    # 孤儿模块
    for p in relations.orphan_modules:
        risky = any(k in p.lower() for k in ('secret', 'key', 'auth', 'gateway', 'crawler'))
        add(
            category='orphan',
            file=p,
            description='孤儿模块：未被其他模块引用',
            evidence='reverse dependency = 0',
            recommendation='确认是否废弃；检查是否遗留敏感逻辑',
            score=55 if risky else 30,
            confidence=0.85,
            risk_reasons=['孤儿模块'] + (['敏感路径'] if risky else []),
            conditions=['orphan'],
        )

    # 重复代码
    for d in relations.duplicate_pairs:
        add(
            category='duplicate',
            file=d.file1,
            description=f"与 {d.file2}::{d.func2} 相似度 {d.similarity}（阈值 0.82）",
            evidence=f"{d.file1}::{d.func1} ~ {d.file2}::{d.func2} sim={d.similarity}",
            recommendation='抽取公共函数消除重复',
            score=int(d.similarity * 60),
            confidence=0.9,
            risk_reasons=['重复代码'],
            conditions=[f'similarity>={0.82}'],
            dependencies=[d.file2],
        )

    # 未处理异常
    for u in relations.unhandled_exceptions:
        add(
            category='exception',
            file=u['file'],
            line=u.get('line'),
            description=u.get('detail', '未处理异常'),
            evidence=f"line {u.get('line')}",
            recommendation='明确捕获异常类型并记录/上抛',
            score=50,
            confidence=0.8,
            risk_reasons=['异常处理不足'],
            conditions=['unhandled_exception'],
        )

    # 活跃 TODO
    for t in relations.todo_priority[:200]:
        marker_score = {'FIXME': 65, 'HACK': 55, 'XXX': 45, 'TODO': 25}.get(t['marker'], 20)
        add(
            category='todo',
            file=t['file'],
            line=t.get('line'),
            description=f"{t['marker']}: {t.get('text', '')[:160]}",
            evidence=t.get('text', ''),
            recommendation='排期处理或降级为文档说明',
            score=marker_score,
            confidence=0.75,
            risk_reasons=['技术债'],
            conditions=['active_todo'],
        )

    # 密钥疑似
    for pf in parsed:
        for s in pf.suspected_secrets:
            add(
                category='secret',
                file=pf.path,
                line=s.get('line'),
                description='疑似硬编码密钥/令牌',
                evidence=s.get('text', ''),
                recommendation='移入环境变量或密钥管理服务，轮换凭据',
                score=95,
                confidence=0.85,
                risk_reasons=['疑似密钥'],
                conditions=['suspected_secret'],
            )

    # 硬编码 URL
    for pf in parsed:
        if not pf.urls:
            continue
        # 每个文件聚合一条
        add(
            category='hardcoded_url',
            file=pf.path,
            description=f"{len(pf.urls)} 处硬编码 URL",
            evidence='; '.join(pf.urls[:5]),
            recommendation='收敛为配置项，避免环境耦合',
            score=35 + min(len(pf.urls), 20),
            confidence=0.8,
            risk_reasons=['硬编码URL'] if len(pf.urls) >= 5 else [],
            conditions=['has_url'],
        )

    # 环境变量引用
    env_files = [pf for pf in parsed if pf.env_refs]
    for pf in env_files:
        add(
            category='env_ref',
            file=pf.path,
            description=f"引用 {len(pf.env_refs)} 个环境变量",
            evidence=', '.join(pf.env_refs[:10]),
            recommendation='文档化必需环境变量并校验缺失场景',
            score=20,
            confidence=0.7,
            conditions=['uses_env'],
        )

    # git 风险：仅 bot 修改的核心文件 + 近期人类提交标记
    recent_paths: Set[str] = set()
    if git and git.available:
        import time as _time
        cutoff = _time.time() - git.since_days * 86400
        for path, st in git.file_stats.items():
            if st.from_bot_only and path in lang_by_path:
                add(
                    category='git_bot_only',
                    file=path,
                    description='近期仅由机器人账号修改',
                    evidence=', '.join(st.authors),
                    recommendation='人工复核机器人提交',
                    score=45,
                    confidence=0.7,
                    risk_reasons=['bot-only'],
                    conditions=['bot_only_commits'],
                    git_authors=st.authors,
                )
            if (not st.from_bot_only and st.last_commit_at >= cutoff
                    and st.commit_count > 0 and path in lang_by_path):
                recent_paths.add(path)

    # 为近期有真实提交的文件补 recent 条件（供 secret+recent 等交集）
    for f in findings:
        if f.file in recent_paths and 'recent_git' not in f.conditions:
            f.conditions.append('recent_git')

    # 条件交集在 decision 层计算后回填 score 修正
    return findings


def apply_condition_intersections(findings: List[Finding]) -> Dict[str, List[str]]:
    """条件交集：同时满足多个条件的文件提升优先级。"""
    by_file: Dict[str, List[Finding]] = defaultdict(list)
    for f in findings:
        if f.file:
            by_file[f.file].append(f)

    intersections: Dict[str, List[str]] = {}

    def condition_of(fset: List[Finding]) -> Set[str]:
        s: Set[str] = set()
        for f in fset:
            s.update(f.conditions)
        return s

    combos = [
        ('complexity+todo', {'complexity>15', 'active_todo'}),
        # 必须同时满足：疑似密钥 + 窗口内真实人类提交
        ('secret+recent', {'suspected_secret', 'recent_git'}),
        ('orphan+secret', {'orphan', 'suspected_secret'}),
        ('cycle+complexity', {'has_cycle', 'complexity>15'}),
        ('dup+complexity', {'similarity>=0.82', 'complexity>15'}),
        ('todo+orphan', {'active_todo', 'orphan'}),
    ]
    for name, needed in combos:
        matched = []
        for fpath, fset in by_file.items():
            conds = condition_of(fset)
            # duplicate 条件以 category 补判（条件串可能被写入时格式化差异）
            if 'similarity>=0.82' in needed and any(f.category == 'duplicate' for f in fset):
                conds = set(conds) | {'similarity>=0.82'}
            if not needed <= conds:
                continue
            matched.append(fpath)
            for f in fset:
                f.conditions.append(f'intersect:{name}')
                f.score += 25
                f.risk_reasons.append(f'条件交集:{name}')
        if matched:
            intersections[name] = sorted(matched)
    return intersections


def prioritize(findings: List[Finding]) -> List[Finding]:
    """按风险分排序并分配 priority。"""
    for f in findings:
        if f.severity == 'HIGH':
            f.score += 20
        elif f.severity == 'MEDIUM':
            f.score += 10
    findings.sort(key=lambda x: (-x.score, x.file, x.line or 0))
    for i, f in enumerate(findings, 1):
        f.priority = i
    return findings


def group_findings(findings: List[Finding]) -> Dict[str, Dict[str, List[str]]]:
    by_lang: Dict[str, List[str]] = defaultdict(list)
    by_dir: Dict[str, List[str]] = defaultdict(list)
    by_arc: Dict[str, List[str]] = defaultdict(list)
    by_risk: Dict[str, List[str]] = defaultdict(list)
    for f in findings:
        by_lang[f.language or 'unknown'].append(f.id)
        by_dir[f.directory].append(f.id)
        by_arc[f.archive or '(filesystem)'].append(f.id)
        by_risk[f.severity].append(f.id)
    return {
        'language': dict(by_lang),
        'directory': dict(by_dir),
        'archive': dict(by_arc),
        'risk': dict(by_risk),
    }


def decide(
    parsed: List[ParsedFile],
    relations: RelationReport,
    git: Optional[GitHistoryResult] = None,
    archive_members: Optional[Set[str]] = None,
) -> DecisionReport:
    rep = DecisionReport()
    findings = build_findings(parsed, relations, git, archive_members)
    rep.condition_intersections = apply_condition_intersections(findings)
    findings = prioritize(findings)
    rep.findings = findings
    groups = group_findings(findings)
    rep.groups_by_language = groups['language']
    rep.groups_by_directory = groups['directory']
    rep.groups_by_archive = groups['archive']
    rep.groups_by_risk = groups['risk']
    rep.topological_order = relations.topological_order

    sev_counts = defaultdict(int)
    for f in findings:
        sev_counts[f.severity] += 1
    rep.summary = {
        'total_findings': len(findings),
        'high': sev_counts.get('HIGH', 0),
        'medium': sev_counts.get('MEDIUM', 0),
        'low': sev_counts.get('LOW', 0),
        'files_analyzed': len(parsed),
        'condition_intersections': len(rep.condition_intersections),
        'languages': len(rep.groups_by_language),
        'directories': len(rep.groups_by_directory),
        'archives': len(rep.groups_by_archive),
    }
    return rep
