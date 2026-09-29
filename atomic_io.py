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
import tempfile
import threading
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
]

# 超过该大小的文件用 mmap 读（字节）。
MMAP_THRESHOLD = 8 * 1024 * 1024

_FSYNC = os.environ.get("AGENTRL_FSYNC", "1") != "0"


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


def _commit(tmp_path: str, final_path: str, do_fsync: bool) -> None:
    """把写好的临时文件原子地搬到最终路径。"""
    os.replace(tmp_path, final_path)
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


def atomic_append_line(path: Union[str, os.PathLike], line: str, *,
                       encoding: str = "utf-8", fsync: bool = True) -> None:
    """追加一行（jsonl 场景）：进程内加锁 + O_APPEND 单次 write + fsync。

    追加语义无法用 os.replace 表达，退而求其次保证「单次写入完整行」，
    避免多行交错。

    注意：Windows 上 ``_O_APPEND`` 的「定位到文件尾」与「写入」并非原子，
    同进程多线程各自持 fd 追加时会互相覆盖（实测 4 线程 ×30 行会丢行）。
    这里用进程内锁把 open→write→fsync→close 串起来；跨进程写入依赖
    单写者队列（见 write_queue），不在此处提供文件锁。
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    data = line.encode(encoding)
    if not data.endswith(b"\n"):
        data += b"\n"
    with _APPEND_LOCK:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)
            if fsync and _FSYNC:
                os.fsync(fd)
        finally:
            os.close(fd)


# ---------------------------------------------------------------- 读接口
def read_bytes(path: Union[str, os.PathLike]) -> bytes:
    """一次性读取整个文件为 bytes。"""
    with open(path, "rb") as fh:
        return fh.read()


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

    size = os.path.getsize(path)
    if size < threshold:
        yield read_bytes(path)
        return

    import mmap
    with open(path, "rb") as fh:
        if size == 0:
            yield b""
            return
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            yield mm
        finally:
            mm.close()


def read_json(path: Union[str, os.PathLike]) -> Any:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


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
