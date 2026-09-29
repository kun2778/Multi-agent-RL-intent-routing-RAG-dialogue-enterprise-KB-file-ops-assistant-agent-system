"""读写锁（RWLock）。

``threading`` 只提供互斥锁，读多写少的共享状态（意图语料索引、Q 网络权重
读取、引擎实例表）用互斥锁会让本可并发的读操作互相排队。

本实现为**写者优先**：一旦有写者在等待，新到的读者排队，
避免写者被连续到来的读者饿死。

用法::

    lock = RWLock()

    with lock.read():        # 多个读者并发
        data = cache[key]

    with lock.write():       # 独占
        cache[key] = value

也可用 ``lock.acquire_read() / release_read()`` 手动配对。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

__all__ = ["RWLock"]


class RWLock:
    """写者优先的读写锁。不可重入（与 threading.Lock 同语义）。"""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    # -------------------------------------------------------------- 读
    def acquire_read(self, timeout: float = -1) -> bool:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: not self._writer and self._waiting_writers == 0,
                None if timeout is None or timeout < 0 else timeout,
            )
            if ok:
                self._readers += 1
            return ok

    def release_read(self) -> None:
        with self._cond:
            if self._readers <= 0:
                raise RuntimeError("release_read() without matching acquire_read()")
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    # -------------------------------------------------------------- 写
    def acquire_write(self, timeout: float = -1) -> bool:
        with self._cond:
            self._waiting_writers += 1
            try:
                ok = self._cond.wait_for(
                    lambda: not self._writer and self._readers == 0,
                    None if timeout is None or timeout < 0 else timeout,
                )
            finally:
                self._waiting_writers -= 1
            if ok:
                self._writer = True
            return ok

    def release_write(self) -> None:
        with self._cond:
            if not self._writer:
                raise RuntimeError("release_write() without matching acquire_write()")
            self._writer = False
            self._cond.notify_all()

    # ------------------------------------------------------ 上下文管理器
    @contextmanager
    def read(self, timeout: float = -1) -> Iterator[None]:
        """共享读临界区。"""
        if not self.acquire_read(timeout):
            raise TimeoutError("acquire_read timed out")
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self, timeout: float = -1) -> Iterator[None]:
        """独占写临界区。"""
        if not self.acquire_write(timeout):
            raise TimeoutError("acquire_write timed out")
        try:
            yield
        finally:
            self.release_write()

    # ------------------------------------------------------ 状态查询
    @property
    def locked_for_write(self) -> bool:
        with self._cond:
            return self._writer

    @property
    def reader_count(self) -> int:
        with self._cond:
            return self._readers

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        with self._cond:
            return (f"<RWLock readers={self._readers} "
                    f"writer={self._writer} waiting_writers={self._waiting_writers}>")
