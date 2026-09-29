"""采集层：项目根/子项目定位、递归遍历、忽略过滤、跳过记录、压缩包内存索引、编码检测。"""

import hashlib
import os
import stat
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from compression_handler import (
    collect_archive_index,
    detect_and_parse_package,
    is_compression_format,
)
from ignore_rules import IgnoreRulesEngine, create_default_engine
from parallel_scanner import EXCLUDED_DIR_NAMES, READABLE_EXTENSIONS


PROJECT_MARKERS = {
    'package.json', 'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt',
    'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', 'CMakeLists.txt',
    'Makefile', '.git', 'composer.json', 'Gemfile', 'mix.exs',
}

MAX_ARCHIVE_SIZE = 50 * 1024 * 1024
MAX_FILE_SIZE = 5 * 1024 * 1024

BINARY_EXTS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.tif', '.tiff',
    '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.class', '.o', '.a',
    '.woff', '.woff2', '.ttf', '.otf', '.eot',
    '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.db', '.sqlite', '.pyc', '.pyo', '.pyd', '.whl', '.egg',
}
# .pdf 不在二进制列表中：走 pdf_parser 专用解析


@dataclass
class SkipRecord:
    path: str
    reason: str
    category: str  # symlink | permission | binary | encoding | size | archive | ignore | other


@dataclass
class FileEntry:
    path: str                 # 相对根路径，压缩包内为 archive!inner
    abs_path: str = ''
    size: int = 0
    mtime: float = 0.0
    language_ext: str = ''
    encoding: str = ''
    readable: bool = False
    in_archive: str = ''
    is_archive: bool = False
    subproject: str = ''
    sha1: str = ''


@dataclass
class CollectionResult:
    root: str
    subprojects: List[str] = field(default_factory=list)
    files: List[FileEntry] = field(default_factory=list)
    skipped: List[SkipRecord] = field(default_factory=list)
    archives: List[Dict] = field(default_factory=list)       # 压缩包元数据
    archive_index: List[Dict] = field(default_factory=list)  # 包内文件索引
    archive_packages: Dict[str, object] = field(default_factory=dict)
    contents: Dict[str, str] = field(default_factory=dict)   # path -> text
    encodings: Dict[str, str] = field(default_factory=dict)
    ignore_sources: List[str] = field(default_factory=list)
    bytes_read: int = 0
    total_walked: int = 0


def detect_encoding(raw: bytes) -> Tuple[Optional[str], bool]:
    """返回 (encoding, is_binary)。无法识别时 encoding=None。"""
    if b'\x00' in raw[:8192]:
        return None, True
    if raw.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig', False
    if raw.startswith(b'\xff\xfe\x00\x00'):
        return 'utf-32-le', False
    if raw.startswith(b'\xff\xfe'):
        return 'utf-16-le', False
    if raw.startswith(b'\xfe\xff'):
        return 'utf-16-be', False
    for enc in ('utf-8', 'gb18030', 'gbk', 'big5', 'shift_jis', 'latin-1'):
        try:
            raw.decode(enc)
            return enc, False
        except (UnicodeDecodeError, LookupError):
            continue
    return None, False


def locate_project_root(path: str) -> str:
    path = os.path.abspath(path)
    if any(os.path.exists(os.path.join(path, m)) for m in PROJECT_MARKERS):
        return path
    # 向上找
    cur = path
    for _ in range(5):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
        if any(os.path.exists(os.path.join(cur, m)) for m in PROJECT_MARKERS):
            return cur
    return path


def find_subprojects(root: str, ignore: Optional[IgnoreRulesEngine] = None,
                     max_depth: int = 2) -> List[str]:
    """定位一级/二级子项目（含项目标记的目录）。"""
    found: List[str] = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == '.' else rel.replace('\\', '/').count('/') + 1
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in {x.lower() for x in EXCLUDED_DIR_NAMES}
            and not d.startswith('.')
        ]
        if dirpath == root:
            continue
        markers = PROJECT_MARKERS & set(filenames)
        # package.json 在 node 子项目等
        if markers - {'.git'}:
            found.append(os.path.relpath(dirpath, root).replace('\\', '/'))
    return sorted(set(found))


def _should_skip_dir(name: str, ignore: IgnoreRulesEngine, rel: str) -> bool:
    if name.lower() in {d.lower() for d in EXCLUDED_DIR_NAMES}:
        return True
    if name.startswith('.') and name not in ('.github',):
        # 仍交给 ignore 引擎决定（.git 必排除）
        if ignore.should_prune_dir(rel) or name in ('.git', '.svn', '.hg'):
            return True
    return ignore.should_prune_dir(rel)


def collect(
    root: str,
    ignore: Optional[IgnoreRulesEngine] = None,
    max_file_size: int = MAX_FILE_SIZE,
    max_archive_size: int = MAX_ARCHIVE_SIZE,
    read_contents: bool = True,
    include_archives: bool = True,
    extra_exclude_dirs: Optional[Iterable[str]] = None,
) -> CollectionResult:
    """只读采集：遍历 + 过滤 + 编码检测 + 压缩包内存解析。"""
    root = os.path.abspath(root)
    project_root = locate_project_root(root)
    if ignore is None:
        ignore = create_default_engine(project_root)
    elif not ignore.get_loaded_sources():
        ignore.load_all()

    result = CollectionResult(root=project_root)
    result.ignore_sources = ignore.get_loaded_sources()
    result.subprojects = find_subprojects(project_root, ignore)

    extra_ex = {e.strip('/').lower() for e in (extra_exclude_dirs or [])}
    sub_set = set(result.subprojects)

    def skip(path: str, reason: str, category: str):
        result.skipped.append(SkipRecord(path=path, reason=reason, category=category))

    def walk(dirpath: str):
        try:
            with os.scandir(dirpath) as it:
                entries = list(it)
        except PermissionError as e:
            skip(os.path.relpath(dirpath, project_root), f"权限不足: {e}", 'permission')
            return
        except OSError as e:
            skip(os.path.relpath(dirpath, project_root), f"无法读取目录: {e}", 'permission')
            return

        for entry in entries:
            try:
                rel = os.path.relpath(entry.path, project_root).replace('\\', '/')
            except ValueError:
                skip(entry.path, '路径不可相对化', 'other')
                continue

            is_symlink = entry.is_symlink()
            if is_symlink:
                skip(rel, '符号链接（跳过防循环）', 'symlink')
                continue

            try:
                if entry.is_dir(follow_symlinks=False):
                    name = entry.name
                    if _should_skip_dir(name, ignore, rel):
                        skip(rel + '/', '目录排除(ignore/system)', 'ignore')
                        continue
                    if extra_ex and name.lower() in extra_ex:
                        skip(rel + '/', '用户指定排除目录', 'ignore')
                        continue
                    walk(entry.path)
                    continue

                if not entry.is_file(follow_symlinks=False):
                    skip(rel, '非普通文件', 'other')
                    continue
            except PermissionError as e:
                skip(rel, f"权限不足: {e}", 'permission')
                continue
            except OSError as e:
                skip(rel, f"stat 失败: {e}", 'permission')
                continue

            if ignore.should_exclude(rel):
                skip(rel, 'ignore 规则排除', 'ignore')
                continue

            try:
                st = entry.stat(follow_symlinks=False)
            except OSError as e:
                skip(rel, f"权限/stat: {e}", 'permission')
                continue

            if st.st_size > max_file_size and not is_compression_format(entry.name):
                skip(rel, f"大小超限 {st.st_size} > {max_file_size}", 'size')
                continue

            ext = os.path.splitext(entry.name)[1].lower()
            if ext in BINARY_EXTS:
                skip(rel, '二进制扩展名', 'binary')
                result.files.append(FileEntry(
                    path=rel, abs_path=entry.path, size=st.st_size,
                    mtime=st.st_mtime, language_ext=ext, readable=False,
                    subproject=_subproject_of(rel, sub_set),
                ))
                continue

            # 压缩包
            if include_archives and is_compression_format(entry.name):
                if st.st_size > max_archive_size:
                    skip(rel, f"压缩包超限 {st.st_size} > {max_archive_size}", 'archive')
                    continue
                _collect_archive(entry.path, rel, st, result, skip, max_file_size,
                                 read_contents=read_contents)
                continue

            # 只读文本读取
            sub = _subproject_of(rel, sub_set)
            fe = FileEntry(
                path=rel, abs_path=entry.path, size=st.st_size,
                mtime=st.st_mtime, language_ext=ext, subproject=sub,
            )
            result.total_walked += 1

            # PDF：走专用解析模块
            if ext == '.pdf':
                if read_contents:
                    text = None
                    try:
                        from pdf_parser import extract_pdf_text
                        text = extract_pdf_text(entry.path, max_chars=max_file_size)
                    except Exception:
                        text = None
                    if text:
                        fe.readable = True
                        fe.encoding = 'pdf'
                        result.encodings[rel] = 'pdf'
                        result.contents[rel] = text
                        result.bytes_read += len(text.encode('utf-8', errors='replace'))
                    else:
                        skip(rel, 'PDF 解析失败或无文本（可能为扫描件）', 'binary')
                result.files.append(fe)
                continue

            if read_contents:
                try:
                    with open(entry.path, 'rb') as fh:
                        raw = fh.read(max_file_size + 1)
                except PermissionError as e:
                    skip(rel, f"权限不足: {e}", 'permission')
                    result.files.append(fe)
                    continue
                except OSError as e:
                    skip(rel, f"读取失败: {e}", 'other')
                    result.files.append(fe)
                    continue

                if len(raw) > max_file_size:
                    skip(rel, f"大小超限 > {max_file_size}", 'size')
                    result.files.append(fe)
                    continue

                enc, is_bin = detect_encoding(raw)
                if is_bin or enc is None:
                    skip(rel, '二进制或编码无法识别', 'binary' if is_bin else 'encoding')
                    result.files.append(fe)
                    continue
                try:
                    text = raw.decode(enc)
                except (UnicodeDecodeError, LookupError) as e:
                    skip(rel, f"编码失败: {e}", 'encoding')
                    result.files.append(fe)
                    continue

                fe.readable = True
                fe.encoding = enc
                result.encodings[rel] = enc
                result.contents[rel] = text
                result.bytes_read += len(raw)

            result.files.append(fe)

    walk(project_root)
    return result


def _subproject_of(rel: str, sub_set: Set[str]) -> str:
    rel = rel.replace('\\', '/')
    best = ''
    for s in sub_set:
        if rel == s or rel.startswith(s + '/'):
            if len(s) > len(best):
                best = s
    return best


def _collect_archive(abs_path: str, rel: str, st, result: CollectionResult,
                     skip, max_file_size: int, read_contents: bool = True):
    pkg = detect_and_parse_package(filepath=abs_path, max_package_size=MAX_ARCHIVE_SIZE,
                                   max_file_size=max_file_size)
    if pkg is None:
        skip(rel, '压缩包读取失败', 'archive')
        return
    if not pkg.is_valid:
        skip(rel, pkg.error_reason or '压缩包无效', 'archive')
        result.archives.append({
            'path': rel, 'size': st.st_size, 'mtime': st.st_mtime,
            'valid': False, 'error': pkg.error_reason,
        })
        return

    for s in pkg.skipped:
        skip(f"{rel}!{s['path']}", s['reason'], 'archive')

    stats = pkg.get_statistics()
    result.archives.append({
        'path': rel,
        'size': st.st_size,
        'mtime': st.st_mtime,
        'valid': True,
        'format': pkg.detected_format,
        'inner_files': stats['total_files'],
        'skipped': stats['skipped_count'],
    })
    index = collect_archive_index(pkg, rel)
    result.archive_index.extend(index)
    result.archive_packages[rel] = pkg

    if not read_contents:
        return

    for inner, info in pkg.get_file_index().items():
        vpath = f"{rel}!{inner}"
        content_bytes = info.get('content') or b''
        if len(content_bytes) > max_file_size:
            skip(vpath, '包内文件超限', 'archive')
            continue
        enc, is_bin = detect_encoding(content_bytes)
        if is_bin or enc is None:
            skip(vpath, '包内二进制/编码无法识别', 'binary' if is_bin else 'encoding')
            result.files.append(FileEntry(
                path=vpath, size=info.get('size', 0),
                mtime=float(info['mtime'][0]) if isinstance(info.get('mtime'), tuple) and info['mtime'] else st.st_mtime,
                in_archive=rel, readable=False,
                language_ext=os.path.splitext(inner)[1].lower(),
                subproject='',
            ))
            continue
        try:
            text = content_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            skip(vpath, '包内解码失败', 'encoding')
            continue
        result.contents[vpath] = text
        result.encodings[vpath] = enc
        result.bytes_read += len(content_bytes)
        mtime = st.st_mtime
        if isinstance(info.get('mtime'), tuple) and len(info['mtime']) >= 3:
            try:
                import datetime
                mtime = datetime.datetime(*info['mtime'][:6]).timestamp()
            except (ValueError, TypeError):
                pass
        elif isinstance(info.get('mtime'), (int, float)):
            mtime = float(info['mtime'])
        result.files.append(FileEntry(
            path=vpath, size=info.get('size', 0), mtime=mtime,
            in_archive=rel, readable=True, encoding=enc,
            language_ext=os.path.splitext(inner)[1].lower(),
            subproject='',
        ))


def content_sha1(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8', errors='replace')).hexdigest()
