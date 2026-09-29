"""
并行文件扫描器：os.scandir + 线程池 + 符号链接检测。

功能：
  - 使用 os.scandir 高效遍历目录（比 os.walk 快 2-5 倍）
  - 线程池并行读取文件内容
  - 符号链接/硬链接检测，防止循环遍历
  - 系统目录排除
  - 文件大小限制
  - 实时进度回调
"""

import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from registry_db import RegistryDB, compute_file_hash

# ──────────────────────────────────────────────────────────────
# 可读文件扩展名（统一定义）
# ──────────────────────────────────────────────────────────────

READABLE_EXTENSIONS = {
    'txt', 'md', 'yaml', 'yml', 'py', 'json', 'xml', 'slnx', 'ini', 'cfg',
    'env', 'csv', 'html', 'js', 'ts', 'c', 'cpp', 'h', 'java', 'go', 'rs',
    'sh', 'bat', 'ps1', 'sql', 'log', 'gitignore', 'toml', 'requirements',
    'jsx', 'tsx', 'vue', 'svelte', 'rb', 'php', 'pl', 'r', 'scala', 'kt',
    'swift', 'm', 'mm', 'dart', 'lua', 'ex', 'exs', 'erl', 'hs', 'ml',
    'fs', 'fsx', 'clj', 'el', 'vim', 'elvish',
    'pdf',  # 走 pdf_parser 专用解析
}

# ──────────────────────────────────────────────────────────────
# 系统目录排除列表（Windows）
# ──────────────────────────────────────────────────────────────

EXCLUDED_DIRS_WINDOWS = {
    # 系统目录
    "C:\\Windows", "C:\\$Recycle.Bin", "C:\\System Volume Information",
    "C:\\ProgramData", "C:\\Recovery", "C:\\$WinREAgent",
    "C:\\Program Files", "C:\\Program Files (x86)",
    "C:\\Program Files\\Common Files", "C:\\Program Files (x86)\\Common Files",
    "C:\\Windows\\WinSxS", "C:\\Windows\\Installer", "C:\\Windows\\SoftwareDistribution",
    # 用户目录下的大型缓存/数据目录
    "C:\\Users\\84950\\AppData\\Local\\Temp",
    "C:\\Users\\84950\\AppData\\Local\\Microsoft\\Windows\\INetCache",
    "C:\\Users\\84950\\AppData\\Local\\Microsoft\\Windows\\Explorer",
    "C:\\Users\\84950\\AppData\\Local\\CrashDumps",
    "C:\\Users\\84950\\AppData\\Local\\D3DSCache",
    "C:\\Users\\84950\\AppData\\Local\\IconCache.db",
    "C:\\Users\\84950\\AppData\\Roaming\\Microsoft\\Windows\\Recent",
}

# 默认排除的目录名（不区分大小写）
EXCLUDED_DIR_NAMES = {
    # 版本控制
    ".git", ".svn", ".hg",
    # Python
    "__pycache__", ".venv", "venv", "env", ".eggs", "*.egg-info",
    # Node.js
    "node_modules", ".npm", "yarn.lock",
    # IDE
    ".vscode", ".idea", ".eclipse",
    # 系统/隐藏
    ".DS_Store", ".Trash", "desktop.ini",
    # 缓存
    ".cache", ".local", ".conda", ".anaconda",
    ".cargo", ".rustup", ".m2", ".gradle",
    # 大型数据目录
    "AppData", "Application Data",
}


@dataclass
class ScanResult:
    """单个文件的扫描结果。"""
    rel_path: str
    abs_path: str
    title: str
    file_type: str
    size: int
    mtime: float
    readable: bool
    content_hash: str = ""
    content: str = ""  # 仅可读文件填充


@dataclass
class ScanStats:
    """扫描统计信息。"""
    total_scanned: int = 0
    total_registered: int = 0
    total_updated: int = 0
    total_deleted: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0
    bytes_scanned: int = 0
    skip_reasons: List[Dict[str, str]] = field(default_factory=list)

    def record_skip(self, path: str, reason: str, category: str = "other"):
        self.total_skipped += 1
        if len(self.skip_reasons) < 5000:
            self.skip_reasons.append({"path": path, "reason": reason, "category": category})


class ParallelScanner:
    """并行文件扫描器。"""

    def __init__(
        self,
        root: str,
        registry: RegistryDB,
        max_workers: int = 8,
        max_file_size: int = 5 * 1024 * 1024,  # 5MB
        compute_hash: bool = False,
        excluded_dirs: Optional[Set[str]] = None,
        excluded_dir_names: Optional[Set[str]] = None,
        readable_extensions: Optional[Set[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Args:
            root: 扫描根目录
            registry: SQLite 注册表实例
            max_workers: 并行读取线程数
            max_file_size: 最大文件大小（字节），超过跳过
            compute_hash: 是否计算文件内容哈希（用于增量扫描，但会增加耗时）
            excluded_dirs: 完全排除的目录路径集合
            excluded_dir_names: 排除的目录名集合（不区分大小写）
            readable_extensions: 可读文件扩展名集合
            progress_callback: 进度回调 (已扫描数, 总数, 当前路径)
        """
        self.root = os.path.abspath(root)
        self.registry = registry
        self.max_workers = max_workers
        self.max_file_size = max_file_size
        self.compute_hash = compute_hash
        self.excluded_dirs = excluded_dirs or EXCLUDED_DIRS_WINDOWS
        self.excluded_dir_names = excluded_dir_names or EXCLUDED_DIR_NAMES
        self.readable_extensions = readable_extensions or READABLE_EXTENSIONS
        self.progress_callback = progress_callback
        self._last_stats: Optional[ScanStats] = None

        # 符号链接/硬链接检测
        self._seen_inodes: Set[Tuple[int, int]] = set()  # (st_dev, st_ino)

    def _is_excluded_dir(self, dirpath: str, dirname: str) -> bool:
        """检查目录是否应排除。"""
        # 检查目录名
        if dirname.lower() in {d.lower() for d in self.excluded_dir_names}:
            return True
        # 检查完整路径
        abs_dir = os.path.join(dirpath, dirname)
        if abs_dir in self.excluded_dirs:
            return True
        # 检查 Windows 路径（不区分大小写）
        abs_dir_lower = abs_dir.lower()
        for excluded in self.excluded_dirs:
            if abs_dir_lower == excluded.lower():
                return True
        return False

    def _is_excluded_file(self, filename: str) -> bool:
        """检查文件是否应排除。"""
        if filename.lower() in {d.lower() for d in self.excluded_dir_names}:
            return True
        return False

    def _check_link(self, path: str) -> bool:
        """检查是否为符号链接或硬链接（检测循环）。

        Returns:
            True 如果是安全的（非链接或首次遇到），False 如果是循环链接
        """
        try:
            stat_result = os.stat(path)
            inode_key = (stat_result.st_dev, stat_result.st_ino)
            if inode_key in self._seen_inodes:
                return False  # 已见过，可能是硬链接循环
            self._seen_inodes.add(inode_key)
            return True
        except OSError:
            return False

    def _is_readable(self, filename: str) -> bool:
        """检查文件扩展名是否可读。"""
        ext = os.path.splitext(filename)[1].lstrip('.').lower()
        return ext in self.readable_extensions

    def _read_text(self, path: str) -> Optional[str]:
        """尝试以多种编码读取文本文件；PDF 走专用解析模块。"""
        # PDF：使用 pdf_parser 提取文本（失败返回 None → 标记为不可读）
        if path.lower().endswith('.pdf'):
            try:
                from pdf_parser import extract_pdf_text
                return extract_pdf_text(path)
            except Exception:
                return None
        try:
            from atomic_io import decode_bytes, open_read
            with open_read(path) as raw:
                return decode_bytes(raw)
        except OSError:
            return None

    def scan(self) -> ScanStats:
        """执行完整扫描。

        Returns:
            扫描统计信息
        """
        start_time = time.time()
        stats = ScanStats()

        # 阶段 1：遍历目录
        file_entries = self._walk_directory(stats)

        # 阶段 2：增量对比
        to_update, to_delete = self._diff_with_registry(file_entries, stats)

        # 阶段 3：并行读取文件并更新注册表
        self._process_files(to_update, to_delete, stats)

        stats.elapsed_seconds = time.time() - start_time
        self._last_stats = stats
        print(f"[Scanner] 扫描完成: {stats.total_scanned}文件, {stats.total_updated}更新, {stats.total_deleted}删除, {stats.total_skipped}跳过, {stats.elapsed_seconds:.1f}s")

        return stats

    def _walk_directory(self, stats: ScanStats) -> Dict[str, dict]:
        """遍历目录，收集文件元数据。

        Returns:
            {rel_path: {abs_path, size, mtime, is_readable}}
        """
        entries = {}
        last_print = [0]  # 用列表避免闭包问题

        def _scan_dir(dirpath: str):
            try:
                with os.scandir(dirpath) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                # 检查是否为符号链接目录
                                if entry.is_symlink():
                                    try:
                                        rel_link = os.path.relpath(entry.path, self.root)
                                    except ValueError:
                                        rel_link = entry.path
                                    stats.record_skip(rel_link, "符号链接目录（跳过防循环）", "symlink")
                                    continue  # 跳过符号链接目录

                                dirname = entry.name
                                if self._is_excluded_dir(dirpath, dirname):
                                    try:
                                        rel_ex = os.path.relpath(entry.path, self.root)
                                    except ValueError:
                                        rel_ex = entry.path
                                    stats.record_skip(rel_ex, "目录排除", "ignore")
                                    continue

                                # 硬链接检测
                                if not self._check_link(entry.path):
                                    try:
                                        rel_hl = os.path.relpath(entry.path, self.root)
                                    except ValueError:
                                        rel_hl = entry.path
                                    stats.record_skip(rel_hl, "重复 inode/硬链接", "symlink")
                                    continue

                                _scan_dir(entry.path)

                            elif entry.is_file(follow_symlinks=False):
                                filename = entry.name
                                if self._is_excluded_file(filename):
                                    try:
                                        rel_fe = os.path.relpath(entry.path, self.root)
                                    except ValueError:
                                        rel_fe = entry.path
                                    stats.record_skip(rel_fe, "文件名排除", "ignore")
                                    continue

                                # 检查文件大小
                                try:
                                    stat_result = entry.stat(follow_symlinks=False)
                                except OSError as e:
                                    stats.total_errors += 1
                                    try:
                                        rel_os = os.path.relpath(entry.path, self.root)
                                    except ValueError:
                                        rel_os = entry.path
                                    stats.record_skip(rel_os, f"stat 失败: {e}", "permission")
                                    continue

                                if stat_result.st_size > self.max_file_size:
                                    try:
                                        rel_sz = os.path.relpath(entry.path, self.root)
                                    except ValueError:
                                        rel_sz = entry.path
                                    stats.record_skip(
                                        rel_sz,
                                        f"大小超限 {stat_result.st_size} > {self.max_file_size}",
                                        "size",
                                    )
                                    continue

                                # 跳过空文件
                                if stat_result.st_size == 0:
                                    continue

                                # 计算相对路径
                                try:
                                    rel_path = os.path.relpath(entry.path, self.root)
                                except ValueError:
                                    continue

                                is_readable = self._is_readable(filename)

                                entries[rel_path] = {
                                    "abs_path": entry.path,
                                    "size": stat_result.st_size,
                                    "mtime": stat_result.st_mtime,
                                    "is_readable": is_readable,
                                }
                                stats.total_scanned += 1
                                stats.bytes_scanned += stat_result.st_size

                                if self.progress_callback and stats.total_scanned % 1000 == 0:
                                    self.progress_callback(stats.total_scanned, 0, rel_path)

                        except (OSError, PermissionError) as e:
                            stats.total_errors += 1
                            try:
                                rel_err = os.path.relpath(entry.path, self.root)
                            except (ValueError, UnboundLocalError):
                                rel_err = getattr(entry, 'path', '?')
                            stats.record_skip(rel_err, f"权限/IO 错误: {e}", "permission")
                            continue

            except (OSError, PermissionError) as e:
                stats.total_errors += 1
                stats.record_skip(dirpath, f"目录权限错误: {e}", "permission")

        _scan_dir(self.root)
        return entries

    def get_last_skip_records(self) -> List[Dict[str, str]]:
        """返回最近一次扫描的跳过记录（含原因）。"""
        return list(getattr(self, "_last_stats", None).skip_reasons) if getattr(self, "_last_stats", None) else []

    def _diff_with_registry(
        self, file_entries: Dict[str, dict], stats: ScanStats
    ) -> Tuple[List[str], Set[str]]:
        """与注册表对比，确定需要更新和删除的文件。

        Returns:
            (to_update_paths, to_delete_paths)
        """
        current_paths = set(file_entries.keys())
        registered_paths = self.registry.get_all_paths()

        # 需要删除的文件（在注册表中但不在文件系统中）
        to_delete = registered_paths - current_paths
        stats.total_deleted = len(to_delete)

        # 需要更新的文件（新增或 mtime/hash 变化）
        to_update = []
        for path in current_paths:
            entry = file_entries[path]
            existing = self.registry.get_file(path)

            if existing is None:
                # 新文件
                to_update.append(path)
            elif existing["mtime"] != entry["mtime"]:
                # mtime 变化
                to_update.append(path)
            # 注意：不做 hash 对比（太慢），只对比 mtime

        stats.total_updated = len(to_update)
        return to_update, to_delete

    def _process_files(
        self, to_update: List[str], to_delete: Set[str], stats: ScanStats
    ):
        """并行读取文件内容并更新注册表。"""
        # 先删除不再存在的文件
        if to_delete:
            self.registry.delete_files(to_delete)

        if not to_update:
            return

        # 获取文件条目信息
        file_entries = {}
        for path in to_update:
            abs_path = os.path.join(self.root, path)
            try:
                stat_result = os.stat(abs_path)
                ext = os.path.splitext(path)[1].lstrip('.').lower()
                file_entries[path] = {
                    "abs_path": abs_path,
                    "size": stat_result.st_size,
                    "mtime": stat_result.st_mtime,
                    "is_readable": self._is_readable(os.path.basename(path)),
                    "ext": ext,
                }
            except OSError:
                stats.total_errors += 1

        # 并行读取文件内容
        readable_files = {p: e for p, e in file_entries.items() if e["is_readable"]}
        contents = {}

        if readable_files:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self._read_text, e["abs_path"]): path
                    for path, e in readable_files.items()
                }
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        content = future.result()
                        if content:
                            contents[path] = content
                    except Exception:
                        stats.total_errors += 1

        # 更新注册表
        for path, entry in file_entries.items():
            title = os.path.splitext(os.path.basename(path))[0]
            content = contents.get(path, "")
            content_hash = ""

            # 仅在启用时计算 hash（用于增量扫描）
            if self.compute_hash and entry["is_readable"] and content:
                content_hash = compute_file_hash(entry["abs_path"])

            self.registry.upsert_file(
                path=path,
                title=title,
                file_type=entry["ext"] or "unknown",
                size=entry["size"],
                mtime=entry["mtime"],
                readable=entry["is_readable"] and bool(content),
                content_hash=content_hash,
            )

            if self.progress_callback and stats.total_updated % 500 == 0:
                self.progress_callback(0, stats.total_updated, path)
