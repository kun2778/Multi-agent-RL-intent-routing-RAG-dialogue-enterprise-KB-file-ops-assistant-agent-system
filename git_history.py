"""Git 历史与作者过滤组件。

功能：
- 最近 N 天（默认 45）git log
- 提交作者、邮箱、提交时间
- 排除 ci-bot、dependabot、renovate 等机器人
- 文件修改时间与压缩包内 mtime 关联
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set


BOT_AUTHOR_PATTERNS = (
    re.compile(r'\bci[-_]?bot\b', re.I),
    re.compile(r'\bdependabot\b', re.I),
    re.compile(r'\brenovate\b', re.I),
    re.compile(r'\bgithub-actions\b', re.I),
    re.compile(r'\bgreenkeeper\b', re.I),
    re.compile(r'\bcodecov\b', re.I),
    re.compile(r'\bpre-commit-ci\b', re.I),
    re.compile(r'^.*\[bot\]$', re.I),
    re.compile(r'\bbot$', re.I),
)


@dataclass
class GitCommit:
    sha: str
    author_name: str
    author_email: str
    committed_at: float
    subject: str
    is_bot: bool = False


@dataclass
class GitFileStat:
    path: str
    last_commit_sha: str = ""
    last_commit_at: float = 0.0
    last_author: str = ""
    last_author_email: str = ""
    commit_count: int = 0
    authors: List[str] = field(default_factory=list)
    from_bot_only: bool = False


@dataclass
class GitHistoryResult:
    available: bool = False
    repo_root: Optional[str] = None
    since_days: int = 45
    commits: List[GitCommit] = field(default_factory=list)
    human_commits: List[GitCommit] = field(default_factory=list)
    bot_commits: List[GitCommit] = field(default_factory=list)
    file_stats: Dict[str, GitFileStat] = field(default_factory=dict)
    skipped_reason: str = ""
    excluded_authors: List[str] = field(default_factory=list)


def is_bot_author(name: str, email: str = "") -> bool:
    text = f"{name} <{email}>"
    return any(p.search(text) for p in BOT_AUTHOR_PATTERNS)


def _run_git(args: List[str], cwd: str, timeout: int = 30) -> Optional[str]:
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            creationflags=creationflags,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.decode("utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_repo_root(path: str) -> Optional[str]:
    out = _run_git(["rev-parse", "--show-toplevel"], path)
    if out and out.strip():
        return out.strip()
    return None


def collect_git_history(
    root: str,
    since_days: int = 45,
    include_bot: bool = False,
    max_commits: int = 2000,
) -> GitHistoryResult:
    """收集最近 since_days 天的提交与文件级统计。"""
    result = GitHistoryResult(since_days=since_days)
    root = os.path.abspath(root)

    repo_root = resolve_repo_root(root)
    if not repo_root:
        result.skipped_reason = "非 git 仓库或 git 不可用"
        return result
    result.available = True
    result.repo_root = repo_root

    since_date = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")
    sep = "\x1f"
    fmt = sep.join(["%H", "%an", "%ae", "%at", "%s"])
    log_out = _run_git(
        ["log", f"--since={since_date}", f"--max-count={max_commits}",
         f"--pretty=format:{fmt}", "--name-only"],
        repo_root,
    )
    if log_out is None:
        # 尝试无 --name-only（首次克隆浅仓库等）
        log_out = _run_git(
            ["log", f"--since={since_date}", f"--max-count={max_commits}",
             f"--pretty=format:{fmt}"],
            repo_root,
        )
        if log_out is None:
            result.skipped_reason = "git log 执行失败"
            return result

    current: Optional[GitCommit] = None
    bot_names: Set[str] = set()
    for line in log_out.splitlines():
        if not line.strip():
            continue
        parts = line.split(sep)
        if len(parts) >= 5:
            sha, an, ae, at, subject = parts[0], parts[1], parts[2], parts[3], sep.join(parts[4:])
            try:
                ts = float(at)
            except ValueError:
                ts = 0.0
            bot = is_bot_author(an, ae)
            current = GitCommit(sha, an, ae, ts, subject, bot)
            result.commits.append(current)
            if bot:
                result.bot_commits.append(current)
                bot_names.add(an)
            else:
                result.human_commits.append(current)
        elif current is not None:
            fpath = line.strip().replace("\\", "/")
            if not fpath:
                continue
            # 相对仓库根
            try:
                rel = os.path.relpath(os.path.join(repo_root, fpath), root).replace("\\", "/")
            except ValueError:
                rel = fpath
            if rel.startswith(".."):
                rel = fpath
            st = result.file_stats.setdefault(rel, GitFileStat(path=rel))
            st.commit_count += 1
            if an not in st.authors:
                st.authors.append(an)
            if current.committed_at >= st.last_commit_at:
                st.last_commit_at = current.committed_at
                st.last_commit_sha = current.sha
                st.last_author = an
                st.last_author_email = ae
            if bot:
                st.from_bot_only = True
            else:
                st.from_bot_only = False

    # from_bot_only 修正：有任何人类提交则 False
    for st in result.file_stats.values():
        if any(not is_bot_author(a) for a in st.authors):
            st.from_bot_only = False

    result.excluded_authors = sorted(bot_names) if not include_bot else []
    if not include_bot:
        result.commits = [c for c in result.commits if not c.is_bot]

    return result


def filter_paths_by_git(
    paths: List[str],
    history: GitHistoryResult,
    mode: str = "recent_human",
    since_days: int = 45,
) -> List[str]:
    """按 git 条件交集过滤路径。

    mode:
      - recent_human: 最近 since_days 内有非 bot 提交（无历史的文件保留在 other）
      - bot_only: 仅 bot 修改
      - any: 有任意提交
    """
    if not history.available:
        return list(paths)
    cutoff = time.time() - since_days * 86400
    kept = []
    for p in paths:
        st = history.file_stats.get(p.replace("\\", "/"))
        if st is None:
            if mode == "recent_human":
                kept.append(p)  # 无历史：新文件/未跟踪，保留
            continue
        if mode == "bot_only":
            if st.from_bot_only:
                kept.append(p)
        elif mode == "any":
            if st.commit_count > 0:
                kept.append(p)
        else:
            if st.last_commit_at >= cutoff and not st.from_bot_only:
                kept.append(p)
            elif st.commit_count == 0:
                kept.append(p)
    return kept


def format_git_summary(history: GitHistoryResult) -> str:
    if not history.available:
        return f"git 不可用: {history.skipped_reason}"
    lines = [
        f"git 仓库: {history.repo_root}",
        f"窗口: 最近 {history.since_days} 天",
        f"提交总数(过滤后): {len(history.commits)}  "
        f"人类: {len(history.human_commits)}  机器人: {len(history.bot_commits)}",
    ]
    if history.excluded_authors:
        lines.append("已排除作者: " + ", ".join(history.excluded_authors))
    lines.append(f"涉及文件: {len(history.file_stats)}")
    return "\n".join(lines)


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    h = collect_git_history(root)
    print(format_git_summary(h))
