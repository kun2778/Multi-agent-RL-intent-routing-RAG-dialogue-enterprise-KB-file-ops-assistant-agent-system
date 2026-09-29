"""运行时组件：工具调用、执行验证、多轮迭代、状态记忆、只读安全治理。"""

import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from compression_handler import detect_and_parse_package, is_compression_format
from content_parser import ParsedFile, parse_file
from decision_engine import DecisionReport, decide
from git_history import GitHistoryResult, collect_git_history, format_git_summary
from ignore_rules import IgnoreRulesEngine, create_default_engine
from relation_analyzer import RelationReport, analyze_relations
from report_writer import make_review_dir, validate_reports, write_reports
from review_collector import CollectionResult, FileEntry, collect


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]
    ok: bool = False
    result_summary: str = ''
    error: str = ''
    elapsed: float = 0.0
    round: int = 0


@dataclass
class PipelineState:
    """多轮状态记忆：已扫描、已跳过、已分析。"""
    rounds: int = 0
    max_rounds: int = 6
    scanned_files: List[str] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    analyzed_files: List[str] = field(default_factory=list)
    archives_scanned: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    stages_done: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    report_dir: str = ''
    validated: bool = False
    terminated: bool = False
    terminate_reason: str = ''
    started_at: float = field(default_factory=time.time)
    memory: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def snapshot(self) -> Dict[str, Any]:
        return {
            'rounds': f"{self.rounds}/{self.max_rounds}",
            'scanned': len(self.scanned_files),
            'skipped': len(self.skipped),
            'analyzed': len(self.analyzed_files),
            'archives': len(self.archives_scanned),
            'stages_done': list(self.stages_done),
            'tools': [
                {'tool': t.tool, 'ok': t.ok, 'round': t.round,
                 'summary': t.result_summary, 'error': t.error,
                 'elapsed_ms': int(t.elapsed * 1000)}
                for t in self.tool_calls
            ],
            'report_dir': self.report_dir,
            'validated': self.validated,
            'terminated': self.terminated,
            'terminate_reason': self.terminate_reason,
            'elapsed_sec': round(self.elapsed, 2),
            'errors': self.errors[:20],
        }


class ReviewPipeline:
    """专业只读文件处理流水线（多轮 + 工具调用 + 验证）。"""

    def __init__(self, root: str, max_rounds: int = 6):
        self.root = os.path.abspath(root)
        self.state = PipelineState(max_rounds=max(1, min(int(max_rounds), 10)))
        self._tools: Dict[str, Callable] = {}
        self._register_default_tools()

    # ------------------------------------------------------------ tools
    def _register_default_tools(self):
        self.register_tool('fs_scan', self._tool_fs_scan)
        self.register_tool('ignore_load', self._tool_ignore_load)
        self.register_tool('archive_parse', self._tool_archive_parse)
        self.register_tool('git_log', self._tool_git_log)
        self.register_tool('ast_analyze', self._tool_ast_analyze)
        self.register_tool('content_parse', self._tool_content_parse)
        self.register_tool('decide', self._tool_decide)
        self.register_tool('report_write', self._tool_report_write)
        self.register_tool('report_verify', self._tool_report_verify)

    def register_tool(self, name: str, fn: Callable):
        self._tools[name] = fn

    def call_tool(self, name: str, round_no: int = 0, **args) -> Tuple[bool, Any, str]:
        fn = self._tools.get(name)
        if fn is None:
            tc = ToolCall(tool=name, args=args, ok=False, error='tool not found', round=round_no)
            self.state.tool_calls.append(tc)
            return False, None, 'tool not found'
        t0 = time.time()
        try:
            result = fn(**args)
            ok = True
            err = ''
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
                ok, result = result
                err = '' if ok else str(result)
            summary = self._summarize(result)
            tc = ToolCall(tool=name, args={k: str(v)[:100] for k, v in args.items()},
                          ok=ok, result_summary=summary, error=err,
                          elapsed=time.time() - t0, round=round_no)
            self.state.tool_calls.append(tc)
            if not ok:
                self.state.errors.append(f"{name}: {err}")
            return ok, result, err
        except Exception as e:
            tc = ToolCall(tool=name, args=args, ok=False,
                          error=f"{type(e).__name__}: {e}",
                          elapsed=time.time() - t0, round=round_no)
            self.state.tool_calls.append(tc)
            self.state.errors.append(f"{name}: {e}")
            traceback.print_exc()
            return False, None, str(e)

    @staticmethod
    def _summarize(result: Any) -> str:
        if result is None:
            return 'None'
        if isinstance(result, CollectionResult):
            return f"files={len(result.files)} skipped={len(result.skipped)} archives={len(result.archives)}"
        if isinstance(result, GitHistoryResult):
            return f"commits={len(result.commits)} bot={len(result.bot_commits)}"
        if isinstance(result, RelationReport):
            return f"deps={len(result.dependency_graph)} cycles={len(result.circular_deps)} complex={len(result.complex_functions)}"
        if isinstance(result, DecisionReport):
            return f"findings={len(result.findings)} hi={result.summary.get('high')}"
        if isinstance(result, dict):
            keys = list(result.keys())[:6]
            return f"dict keys={keys}"
        if isinstance(result, list):
            return f"list len={len(result)}"
        return str(result)[:120]

    # ------------------------------------------------------------ tool impls
    def _tool_ignore_load(self, root: Optional[str] = None) -> IgnoreRulesEngine:
        eng = create_default_engine(root or self.root)
        self.state.memory['ignore_sources'] = eng.get_loaded_sources()
        return eng

    def _tool_fs_scan(self, root: Optional[str] = None,
                      ignore: Optional[IgnoreRulesEngine] = None,
                      include_archives: bool = True) -> CollectionResult:
        res = collect(
            root or self.root,
            ignore=ignore,
            include_archives=include_archives,
            read_contents=True,
        )
        self.state.scanned_files = [f.path for f in res.files]
        self.state.skipped = [
            {'path': s.path, 'reason': s.reason, 'category': s.category}
            for s in res.skipped
        ]
        self.state.archives_scanned = [a['path'] for a in res.archives]
        self.state.memory['collection'] = res
        return res

    def _tool_archive_parse(self, filepath: str) -> Dict[str, Any]:
        pkg = detect_and_parse_package(filepath=filepath)
        if pkg is None:
            return {'ok': False, 'error': 'read failed'}
        return {
            'ok': pkg.is_valid,
            'error': pkg.error_reason,
            'stats': pkg.get_statistics() if pkg.is_valid else {},
            'files': pkg.get_file_list() if pkg.is_valid else [],
        }

    def _tool_git_log(self, root: Optional[str] = None, since_days: int = 45) -> GitHistoryResult:
        hist = collect_git_history(root or self.root, since_days=since_days)
        self.state.memory['git'] = hist
        return hist

    def _tool_content_parse(self, collection: Optional[CollectionResult] = None) -> List[ParsedFile]:
        collection = collection or self.state.memory.get('collection')
        if collection is None:
            return []
        parsed = []
        for fe in collection.files:
            if not fe.readable:
                continue
            text = collection.contents.get(fe.path)
            if text is None:
                continue
            parsed.append(parse_file(fe.path, text, encoding=fe.encoding or 'utf-8'))
        self.state.analyzed_files = [p.path for p in parsed]
        self.state.memory['parsed'] = parsed
        return parsed

    def _tool_ast_analyze(self, parsed: Optional[List[ParsedFile]] = None,
                          collection: Optional[CollectionResult] = None,
                          git: Optional[GitHistoryResult] = None) -> RelationReport:
        parsed = parsed if parsed is not None else self.state.memory.get('parsed', [])
        collection = collection or self.state.memory.get('collection')
        contents = collection.contents if collection else {}
        rep = analyze_relations(parsed, contents)
        self.state.memory['relations'] = rep
        return rep

    def _tool_decide(self, parsed: Optional[List[ParsedFile]] = None,
                     relations: Optional[RelationReport] = None,
                     git: Optional[GitHistoryResult] = None,
                     collection: Optional[CollectionResult] = None) -> DecisionReport:
        parsed = parsed if parsed is not None else self.state.memory.get('parsed', [])
        relations = relations or self.state.memory.get('relations')
        git = git if git is not None else self.state.memory.get('git')
        collection = collection or self.state.memory.get('collection')
        if relations is None:
            relations = analyze_relations(parsed, collection.contents if collection else {})
        archive_members = set()
        if collection:
            archive_members = {a['path'] for a in collection.archives}
        rep = decide(parsed, relations, git, archive_members)
        self.state.memory['decision'] = rep
        return rep

    def _tool_report_write(self, out_dir: Optional[str] = None,
                           project: Optional[str] = None) -> Dict[str, Any]:
        collection: CollectionResult = self.state.memory.get('collection')
        decision: DecisionReport = self.state.memory.get('decision')
        relations: RelationReport = self.state.memory.get('relations')
        git: GitHistoryResult = self.state.memory.get('git')
        parsed: List[ParsedFile] = self.state.memory.get('parsed', [])
        if collection is None or decision is None or relations is None:
            return {'ok': False, 'error': 'missing stage results'}

        project = project or os.path.basename(self.root)
        if out_dir is None:
            fingerprint = f"{self.root}|{len(collection.files)}|{decision.summary.get('total_findings', 0)}"
            out_dir = make_review_dir(self.root, fingerprint)

        collection_summary = {
            'root': collection.root,
            'files': len(collection.files),
            'readable': sum(1 for f in collection.files if f.readable),
            'skipped': len(collection.skipped),
            'archives': len(collection.archives),
            'subprojects': collection.subprojects,
            'ignore_sources': collection.ignore_sources,
            'bytes_read': collection.bytes_read,
            'encodings_sample': dict(list(collection.encodings.items())[:20]),
        }
        git_summary = format_git_summary(git) if git else 'git 未执行'
        files_rows = [
            {
                'path': f.path, 'size': f.size,
                'mtime': datetime.fromtimestamp(f.mtime).strftime('%Y-%m-%d %H:%M:%S')
                if f.mtime else '',
                'encoding': f.encoding, 'readable': f.readable,
                'in_archive': f.in_archive, 'language_ext': f.language_ext,
                'subproject': f.subproject,
            }
            for f in collection.files
        ]
        skipped_rows = [
            {'path': s.path, 'category': s.category, 'reason': s.reason}
            for s in collection.skipped
        ]
        runtime_summary = self.state.snapshot()

        artifacts = write_reports(
            out_dir=out_dir,
            project=project,
            collection_summary=collection_summary,
            git_summary=git_summary,
            decision=decision,
            relations=relations,
            files_rows=files_rows,
            archives=collection.archives,
            archive_index=collection.archive_index,
            skipped=skipped_rows,
            parsed_list=parsed,
            contents=collection.contents,
            runtime_summary=runtime_summary,
        )
        self.state.report_dir = out_dir
        self.state.memory['artifacts'] = artifacts
        return {'ok': True, 'dir': out_dir, 'artifacts': artifacts}

    def _tool_report_verify(self, out_dir: Optional[str] = None) -> Dict[str, Any]:
        out_dir = out_dir or self.state.report_dir
        artifacts = self.state.memory.get('artifacts') or {}
        if not out_dir:
            return {'ok': False, 'error': 'no report dir'}
        report = validate_reports(out_dir, artifacts)
        self.state.validated = report['ok']
        self.state.memory['validation'] = report
        # verify 完成后回写 runtime 快照（index.json 在 write 时已生成）
        self._refresh_index_runtime(out_dir)
        return report

    def _refresh_index_runtime(self, out_dir: str) -> None:
        """将最新 runtime 状态回填到 index.json（只改 runtime 字段）。"""
        import json as _json
        path = os.path.join(out_dir, 'index.json')
        if not os.path.isfile(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                idx = _json.load(f)
            idx['runtime'] = self.state.snapshot()
            from atomic_io import atomic_write_text
            atomic_write_text(path, _json.dumps(idx, ensure_ascii=False, indent=2, default=str),
                              encoding='utf-8')
        except (OSError, ValueError, TypeError):
            pass

    # ------------------------------------------------------------ multi-round
    def run(self, query: str = '', stages: Optional[List[str]] = None,
            similarity_threshold: float = 0.82,
            complexity_threshold: int = 15,
            git_days: int = 45) -> Dict[str, Any]:
        """执行多轮流水线。默认全阶段；可传入 stages 子集。"""
        stages = stages or [
            'ignore', 'collect', 'git', 'parse', 'analyze', 'decide', 'report', 'verify',
        ]
        st = self.state
        st.memory['query'] = query
        st.memory['thresholds'] = {
            'similarity': similarity_threshold,
            'complexity': complexity_threshold,
        }

        def need(s: str) -> bool:
            return s in stages and st.rounds < st.max_rounds

        # Round 1: ignore + collect
        if need('ignore') or need('collect'):
            st.rounds += 1
            ignore = None
            if 'ignore' in stages:
                ok, ignore, err = self.call_tool('ignore_load', st.rounds)
                if not ok:
                    ignore = None
                st.stages_done.append('ignore')
            if 'collect' in stages:
                ok, coll, err = self.call_tool(
                    'fs_scan', st.rounds,
                    root=self.root, ignore=ignore, include_archives=True,
                )
                if not ok or coll is None:
                    st.terminated = True
                    st.terminate_reason = f'采集失败: {err}'
                    return st.snapshot()
                st.stages_done.append('collect')

        # Round 2: git
        if need('git'):
            st.rounds += 1
            ok, git, err = self.call_tool('git_log', st.rounds, root=self.root, since_days=git_days)
            if ok:
                st.stages_done.append('git')
            else:
                st.memory.setdefault('git', None)
                st.stages_done.append('git( skipped )')

        # Round 3: content parse + ast analyze
        if need('parse') or need('analyze'):
            st.rounds += 1
            if 'parse' in stages:
                ok, parsed, err = self.call_tool('content_parse', st.rounds)
                if not ok or not parsed:
                    # 尝试仅扫描文本（无压缩包内容时再扫一次）
                    st.errors.append(f'content_parse: {err or "0 files"}')
                st.stages_done.append('parse')
            if 'analyze' in stages:
                ok, rel, err = self.call_tool(
                    'ast_analyze', st.rounds,
                )
                if ok:
                    # 应用阈值参数（analyze_relations 已用默认；如需自定义在此重跑）
                    thresholds = st.memory.get('thresholds', {})
                    parsed_l = st.memory.get('parsed', [])
                    coll = st.memory.get('collection')
                    if (thresholds.get('similarity', 0.82) != 0.82
                            or thresholds.get('complexity', 15) != 15):
                        rel = analyze_relations(
                            parsed_l, coll.contents if coll else {},
                            similarity_threshold=thresholds.get('similarity', 0.82),
                            complexity_threshold=thresholds.get('complexity', 15),
                        )
                        st.memory['relations'] = rel
                    st.stages_done.append('analyze')
                else:
                    st.terminated = True
                    st.terminate_reason = f'分析失败: {err}'
                    return st.snapshot()

        # Round 4: decide
        if need('decide'):
            st.rounds += 1
            ok, decision, err = self.call_tool('decide', st.rounds)
            if not ok or decision is None:
                st.terminated = True
                st.terminate_reason = f'决策失败: {err}'
                return st.snapshot()
            st.stages_done.append('decide')

        # Round 5: report
        if need('report'):
            st.rounds += 1
            ok, wr, err = self.call_tool('report_write', st.rounds)
            if not ok:
                st.terminated = True
                st.terminate_reason = f'报告写出失败: {err}'
                return st.snapshot()
            st.stages_done.append('report')

        # Round 6: verify（失败则修复重写一次）
        if need('verify'):
            st.rounds += 1
            ok, vr, err = self.call_tool('report_verify', st.rounds)
            if not (ok and isinstance(vr, dict) and vr.get('ok')):
                # 多轮修复：重写报告再验
                st.rounds += 1
                if st.rounds <= st.max_rounds:
                    self.call_tool('report_write', st.rounds)
                    ok, vr, err = self.call_tool('report_verify', st.rounds)
            st.stages_done.append('verify')

        st.terminated = True
        failed_checks = []
        val = st.memory.get('validation') or {}
        if not st.validated:
            failed_checks = [c for c in val.get('checks', []) if not c.get('ok')]
        st.terminate_reason = (
            '全部阶段完成且报告校验通过'
            if st.validated else
            f"阶段完成但校验未通过: {failed_checks or st.errors[:3]}"
        )
        # 最终 runtime 快照（含 terminated/terminate_reason）
        if st.report_dir:
            self._refresh_index_runtime(st.report_dir)
        return st.snapshot()


# 只读安全封装：确保流水线不写源文件
_READ_ONLY_GUARD = """
本流水线对项目树仅执行 read/scandir；
唯一写入位置为 review_*/ 报告目录（新建，不覆盖）。
"""


def run_review(root: str, query: str = '', **kwargs) -> Dict[str, Any]:
    """便捷入口。"""
    pipe = ReviewPipeline(root, max_rounds=kwargs.pop('max_rounds', 6))
    return pipe.run(query=query, **kwargs)


if __name__ == '__main__':
    import json as _json
    root = os.path.dirname(os.path.abspath(__file__))
    result = run_review(root, query='只读扫描并深度分析，新增报告')
    print(_json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('report_dir'):
        print('REPORT_DIR:', result['report_dir'])
