"""原子文件写入与大文件安全读取工具。

解决两类问题：
  1. 写 —— 进程崩溃 / 断电时读者看到半个文件。
     统一走「临时文件 + fsync + os.replace 同目录原子替换 + 目录 fsync」。
  2. 读 —— 大文件全量 read() 进内存。
     超过阈值改用只读 mmap，小文件仍返回 bytes。

约定：
  - 所有会落盘的状态文件（JSON/模型 checkpoint/索引 meta）一律用本模块写出。
  - 追加型 jsonl（append-only 日志）用 atomic_append_line：单次 write + O_APPEND。
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Union

__all__ = [
    "MMAP_THRESHOLD",
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "atomic_torch_save",
    "atomic_savez",
    "atomic_append_line",
    "open_read",
    "read_bytes",
    "read_json",
    "decode_bytes",
    "open_blob",
    "fsync_dir",
    "sweep_orphan_tmp",
]

# 超过该大小的文件用 mmap 读（字节）。
MMAP_THRESHOLD = 8 * 1024 * 1024

_FSYNC = os.environ.get("AGENTRL_FSYNC", "1") != "0"

# Windows 上 os.replace / open 会因共享冲突间歇失败（ERROR_SHARING_VIOLATION
# 映射为 PermissionError），需有限重试。次数与退避可由环境变量调优。
_RETRY_TIMES = max(1, int(os.environ.get("AGENTRL_IO_RETRY_TIMES", "10")))
_RETRY_DELAY = float(os.environ.get("AGENTRL_IO_RETRY_DELAY", "0.002"))
_RETRYABLE_ERRORS = (PermissionError,)

# 落盘（_commit）单独一档：并发 ReplaceFileW 会互相返回
# ERROR_UNABLE_TO_MOVE_REPLACEMENT{,_2} / UNABLE_TO_REMOVE_REPLACED
# （WinError 1175/1176/1177，映射为 OSError 而非 PermissionError），
# 实测 8 写者 ×2 读者下约半数尝试会失败，必须给足够多次尝试。
_COMMIT_TIMES = max(1, int(os.environ.get("AGENTRL_COMMIT_RETRY_TIMES", "25")))
_COMMIT_DELAY = float(os.environ.get("AGENTRL_COMMIT_RETRY_DELAY", "0.001"))


def _retry_io(fn, *, times: int = _RETRY_TIMES, delay: float = _RETRY_DELAY,
              errors: tuple = _RETRYABLE_ERRORS):
    """对瞬时错误做有限次线性退避重试，最后一次失败原样抛出。"""
    last: Optional[Exception] = None
    for attempt in range(times):
        try:
            return fn()
        except errors as exc:
            last = exc
            if attempt + 1 >= times:
                break
            time.sleep(delay * (attempt + 1))
    assert last is not None
    raise last


# ---------------------------------------------------------------- 基础设施
# 同进程内追加写的互斥（见 atomic_append_line 的说明）
_APPEND_LOCK = threading.Lock()


def fsync_dir(path: Union[str, os.PathLike]) -> None:
    """fsync 目录项，保证 rename 产生的目录项本身也落盘。

    POSIX 下有效；Windows 不允许以 O_RDONLY 打开目录做 fsync，直接跳过
    （os.replace 在 NTFS 上已由系统保证元数据原子性）。
    """
    if os.name == "nt":
        return
    fd = None
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY)
        os.fsync(fd)
    except OSError:
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _replace_file(tmp_path: str, final_path: str) -> None:
    """把临时文件搬到最终路径（Windows 上用 ReplaceFileW）。``os.replace`` 底层是
    ``MoveFileEx(MOVEFILE_REPLACE_EXISTING)``，**只要目标文件存在任意打开句柄就会报
    WinError 5**（即使句柄带了 FILE_SHARE_DELETE），因此「读者越多、原子写越容易失败」。

    ``ReplaceFileW`` 则允许目标被 ``FILE_SHARE_DELETE`` 句柄持有——配合
    :func:`_open_shared` 的读者，读写可以真正并行。目标不存在时
    （首次创建）以及 ReplaceFileW 失败时回退到 ``os.replace``。
    """
    if os.name != "nt":
        os.replace(tmp_path, final_path)
        return
    if not os.path.exists(final_path):
        os.replace(tmp_path, final_path)
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        replace = k32.ReplaceFileW
        replace.restype = wintypes.BOOL
        replace.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                            wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID)
        IGNORE_MERGE = 0x02   # REPLACEFILE_IGNORE_MERGE_ERRORS
        IGNORE_ACL = 0x04     # REPLACEFILE_IGNORE_ACL_ERRORS
        if replace(os.path.abspath(final_path), os.path.abspath(tmp_path),
                   None, IGNORE_MERGE | IGNORE_ACL, None, None):
            return
    except Exception:
        pass
    os.replace(tmp_path, final_path)


def _commit(tmp_path: str, final_path: str, do_fsync: bool) -> None:
    """把写好的临时文件原子地搬到最终路径。

    Windows 下目标文件若正被读者打开，MoveFileEx 会报共享冲突；并发
    ReplaceFileW 之间又会互相返回 1175/1176/1177。因此优先 ReplaceFileW，
    整体在 OSError 上做有界退避重试。
    """
    _retry_io(lambda: _replace_file(tmp_path, final_path),
              times=_COMMIT_TIMES, delay=_COMMIT_DELAY, errors=(OSError,))
    if do_fsync:
        fsync_dir(os.path.dirname(os.path.abspath(final_path)))


def _write_atomic(path: Union[str, os.PathLike], payload: bytes, *,
                  fsync: bool = True, newline_safe: bool = False) -> None:
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    do_fsync = fsync and _FSYNC

    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=directory or None,
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            if do_fsync:
                os.fsync(fh.fileno())
        _commit(tmp_path, path, do_fsync)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- 写接口
def atomic_write_bytes(path: Union[str, os.PathLike], data: bytes, *,
                       fsync: bool = True) -> None:
    """以原子方式写出 bytes。"""
    _write_atomic(path, data, fsync=fsync)


def atomic_write_text(path: Union[str, os.PathLike], text: str, *,
                      encoding: str = "utf-8", fsync: bool = True) -> None:
    """以原子方式写出文本。"""
    _write_atomic(path, text.encode(encoding), fsync=fsync)


def atomic_write_json(path: Union[str, os.PathLike], obj: Any, *,
                      indent: int = 2, ensure_ascii: bool = False,
                      fsync: bool = True, **dump_kwargs: Any) -> None:
    """以原子方式写出 JSON —— 替代 ``json.dump(obj, open(path,'w'))``。"""
    payload = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii,
                         **dump_kwargs).encode("utf-8")
    _write_atomic(path, payload, fsync=fsync)


def atomic_savez(path: Union[str, os.PathLike], *, compressed: bool = True,
                 fsync: bool = True, **arrays: Any) -> None:
    """原子地写出 .npz（临时文件必须以 .npz 结尾，否则 numpy 会再补后缀）。"""
    import numpy as np

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    do_fsync = fsync and _FSYNC

    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".npz", dir=directory or None,
    )
    os.close(fd)
    try:
        saver = np.savez_compressed if compressed else np.savez
        saver(tmp_path, **arrays)
        if do_fsync:
            # Windows 的 FlushFileBuffers 要求写权限，只读句柄会 EBADF
            with open(tmp_path, "r+b") as fh:
                fh.flush()
                os.fsync(fh.fileno())
        _commit(tmp_path, path, do_fsync)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_torch_save(path: Union[str, os.PathLike], obj: Any, *,
                      fsync: bool = True) -> None:
    """以原子方式保存 torch checkpoint（先写临时文件再 replace）。"""
    import torch  # 局部导入：非训练环境不必承担 torch 导入成本

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    do_fsync = fsync and _FSYNC

    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=directory or None,
    )
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        if do_fsync:
            # Windows 的 FlushFileBuffers 要求写权限，只读句柄会 EBADF
            with open(tmp_path, "r+b") as fh:
                fh.flush()
                os.fsync(fh.fileno())
        _commit(tmp_path, path, do_fsync)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _append_lock_path(path: str) -> str:
    """跨进程追加锁所在文件（按数据文件绝对路径 hash，放在临时目录，
    避免在 workspace 里留下杂物）。"""
    import hashlib
    key = hashlib.sha1(os.path.abspath(path).encode("utf-8", "ignore")).hexdigest()[:24]
    return os.path.join(tempfile.gettempdir(), f"agentrl_append_{key}.lock")


@contextmanager
def _interprocess_append_lock(path: str) -> Iterator[None]:
    """跨进程排他锁（尽力而为，失败静默降级为旧行为）。

    POSIX 下 ``O_APPEND`` 的单次 ``write`` 天然原子，无需加锁。
    Windows 下 CRT 把「定位到尾」与「写入」拆成两步，多进程并发会互相
    覆盖，必须串行化。字节锁不能加在数据文件上（LockFile 对读取同样生效，
    会让并发读者报 ERROR_LOCK_VIOLATION），因此锁在一个临时目录下的
    旁路锁文件上。
    """
    if os.name != "nt":
        yield
        return
    import msvcrt

    fd = None
    locked = False
    try:
        try:
            fd = os.open(_append_lock_path(path), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            fd = None            # 锁文件都建不出来 -> 降级为无锁（不阻断写入）
        if fd is not None:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                locked = True
            except OSError:
                locked = False   # 降级：至少不比修复前差
        yield
    finally:
        if fd is not None:
            if locked:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            try:
                os.close(fd)
            except OSError:
                pass


def atomic_append_line(path: Union[str, os.PathLike], line: str, *,
                       encoding: str = "utf-8", fsync: bool = True) -> None:
    """追加一行（jsonl 场景）：进程内锁 + 跨进程锁 + O_APPEND 单次 write + fsync。

    追加语义无法用 os.replace 表达，退而求其次保证「单次写入完整行」，
    避免多行交错。

    注意：Windows 上 ``_O_APPEND`` 的「定位到文件尾」与「写入」并非原子，
    同进程多线程各自持 fd 追加时会互相覆盖（实测 4 线程 ×30 行会丢行），
    这里用 ``_APPEND_LOCK`` 把 open→write→fsync→close 串起来；跨进程则
    由 ``_interprocess_append_lock`` 提供旁路文件锁（POSIX 无需）。
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    data = line.encode(encoding)
    if not data.endswith(b"\n"):
        data += b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        # 缺 O_BINARY 会让 MSVC CRT 把 \n 翻成 \r\n，与 atomic_write_text 不一致
        flags |= os.O_BINARY
    with _APPEND_LOCK:
        with _interprocess_append_lock(path):
            fd = _retry_io(lambda: os.open(path, flags, 0o644))
            try:
                os.write(fd, data)
                if fsync and _FSYNC:
                    os.fsync(fd)
            finally:
                os.close(fd)


# ---------------------------------------------------------------- 读接口
def _open_shared(path: Union[str, os.PathLike], mode: str = "rb"):
    """打开文件用于读取，Windows 下附带 ``FILE_SHARE_DELETE``。

    Python 的 ``open()`` / ``os.open()`` 底层用 ``_SH_DENYNO``
    （只共享 READ|WRITE），读句柄会让并发的 ``os.replace`` 报
    ``PermissionError [WinError 5]``，从而击穿「原子写」的承诺
    （读者越多，写者越容易失败）。改用 ``CreateFileW`` 带
    ``FILE_SHARE_READ|WRITE|DELETE`` 后，读者不再挡住原子替换。

    任何一步失败都退回普通 ``open()``，异常语义与之前完全一致。
    """
    if os.name != "nt" or mode != "rb":
        return open(path, mode)
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = k32.CreateFileW
        create.restype = wintypes.HANDLE
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE)
        GENERIC_READ = 0x80000000
        FILE_SHARE_ALL = 0x01 | 0x02 | 0x04   # READ | WRITE | DELETE
        OPEN_EXISTING = 3
        FILE_ATTRIBUTE_NORMAL = 0x80
        handle = create(os.fspath(path), GENERIC_READ, FILE_SHARE_ALL, None,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
        if handle is None or handle == wintypes.HANDLE(-1).value:
            return open(path, mode)   # 交给 open 抛出准确的 OSError 子类
        try:
            fd = msvcrt.open_osfhandle(int(handle),
                                       os.O_RDONLY | getattr(os, "O_BINARY", 0))
        except Exception:
            k32.CloseHandle(handle)
            return open(path, mode)
        return os.fdopen(fd, mode)
    except Exception:
        return open(path, mode)


def read_bytes(path: Union[str, os.PathLike]) -> bytes:
    """一次性读取整个文件为 bytes。

    Windows 下与并发写者的 ``os.replace`` 竞争时 open 可能短暂报共享冲突，
    这里重试以兑现「读者永远看到完整文件」的承诺。
    """
    def _do() -> bytes:
        with _open_shared(path) as fh:
            return fh.read()
    return _retry_io(_do)


@contextmanager
def open_read(path: Union[str, os.PathLike], *,
              prefer_mmap: bool = True,
              threshold: int = MMAP_THRESHOLD) -> Iterator[Union[bytes, Any]]:
    """按大小选择读取方式的上下文管理器。

    - 小于 ``threshold``：返回 ``bytes``（全量入内存，复制代价低）
    - 大于等于 ``threshold``：返回只读 ``mmap.mmap``（零拷贝，按需缺页）

    两者都支持 ``len()`` / 索引 / 切片，调用方通常可无感切换。
    mmap 对象必须在 with 块内使用，退出时自动关闭。
    """
    path = os.fspath(path)
    if not prefer_mmap:
        yield read_bytes(path)
        return

    size = _retry_io(lambda: os.path.getsize(path))
    if size < threshold:
        yield read_bytes(path)
        return

    import mmap

    fh = _retry_io(lambda: _open_shared(path))
    try:
        if size == 0:
            yield b""
            return
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            yield mm
        finally:
            mm.close()
    finally:
        fh.close()


def read_json(path: Union[str, os.PathLike]) -> Any:
    """读取 JSON 文件。"""
    import io as _io

    def _do() -> Any:
        with _open_shared(path) as raw:
            with _io.TextIOWrapper(raw, encoding="utf-8") as fh:
                obj = json.load(fh)
        return obj
    return _retry_io(_do)


def decode_bytes(buf: Any,
                 encodings: tuple = ("utf-8", "gbk", "gb18030", "latin-1"),
                 fallback: str = "latin-1") -> str:
    """按候选编码解码二进制内容。

    ``buf`` 既可以是 ``bytes``，也可以是 ``open_read`` 返回的只读 mmap ——
    增量解码器走 buffer protocol，两者都能直接喂进去，
    避免为了调用 ``bytes.decode()`` 而把整个 mmap 复制成 bytes。
    """
    import codecs
    for enc in encodings:
        try:
            return codecs.getincrementaldecoder(enc)(errors="strict").decode(buf, True)
        except (UnicodeDecodeError, ValueError, LookupError):
            continue
    return codecs.getincrementaldecoder(fallback)(errors="ignore").decode(buf, True)


@contextmanager
def open_blob(data: bytes, *, threshold: int = MMAP_THRESHOLD,
              suffix: str = ".blob") -> Iterator[Any]:
    """按大小选择承载方式的可 seek 二进制流（BytesIO 分级）。

    - ``len(data) <= threshold``：直接 ``io.BytesIO``（无磁盘 IO，复制代价小）
    - 更大：落到临时文件再只读打开。``io.BytesIO(b)`` 会复制一份缓冲区，
      对 50MB 的压缩包等于凭空多吃 50MB 常驻内存；临时文件把这份拷贝
      换成磁盘占用，且 tarfile/zipfile 都只要一个可 seek 的对象。

    退出 with 块时自动删除临时文件。
    """
    import io as _io

    if len(data) <= threshold:
        yield _io.BytesIO(data)
        return

    fd, tmp_path = tempfile.mkstemp(prefix="agentrl_blob_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            if _FSYNC:
                os.fsync(fh.fileno())
        with open(tmp_path, "rb") as fh:
            yield fh
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def sweep_orphan_tmp(directory: Union[str, os.PathLike], *,
                     max_age: float = 3600.0,
                     suffixes: tuple = (".tmp",),
                     max_entries: int = 20000) -> int:
    """清理崩溃（kill -9 / 断电）遗留的原子写临时文件。

    ``_write_atomic`` 的 ``except BaseException`` 清理在进程被强杀时不会执行，
    会在 workspace 留下 ``<name>.XXXXXX.tmp`` 孤儿文件。这里按后缀 + 年龄
    （默认 1 小时，远大于任何一次正常写入的耗时）识别，避免误删正在写的
    文件；返回删除数量。目录不存在时返回 0。
    """
    if not directory or not os.path.isdir(directory):
        return 0
    now = time.time()
    removed = 0
    seen = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            seen += 1
            if seen > max_entries:
                return removed
            if not fn.endswith(suffixes):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if now - st.st_mtime < max_age:
                continue
            try:
                os.unlink(fp)
                removed += 1
            except OSError:
                pass
    return removed
