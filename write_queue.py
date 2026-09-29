"""写操作队列化：单写者线程消费，把落盘串行化。

背景：``chat_server`` / ``continuous_learning`` / ``safety`` 等多个线程都会
写状态文件。若各自 ``open('w') + json.dump``，既有锁竞争，又可能互相覆盖。
收敛到一个队列 + 单个消费者线程后：

  - 所有落盘串行，天然消除同一路径的写-写竞争（与 atomic_io 配合，
    读者永远看到完整文件）
  - 提交方只做 ``put``，不阻塞在 IO 上
  - ``flush()`` / ``close()`` 可等待队列排空，便于测试与退出前收尾

用法::

    wq = WriteQueue()
    wq.submit_json(path, obj)          # 异步原子写
    wq.submit_append_line(path, line)  # 异步追加
    wq.flush()                          # 等待当前队列清空
    wq.close()                          # 停线程（幂等）
"""

from __future__ import annotations

import atexit
import os
import queue
import threading
from typing import Any, Callable, Optional, Union

from atomic_io import atomic_append_line, atomic_write_json, atomic_write_text

__all__ = ["WriteQueue", "get_default_write_queue"]

_SENTINEL = object()


class WriteQueue:
    """单写者消费的任务队列。线程安全、惰性启动、close 幂等。

    「已排空」用未完成任务计数判断（而非队列是否为空），
    避免「消费者刚置位 → 提交方清位 → 入队」之间的竞态。
    """

    def __init__(self, name: str = "agentrl-writer") -> None:
        self._name = name
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._drained = threading.Condition(threading.Lock())
        self._outstanding = 0
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._errors: list = []

    # ------------------------------------------------------------ 提交
    def submit(self, fn: Callable[[], Any]) -> None:
        """提交一个无参可调用对象，由写线程执行。"""
        with self._drained:
            if self._closed:
                raise RuntimeError(f"{self._name} is closed")
            self._outstanding += 1
            self._ensure_thread_locked()
        self._queue.put(fn)

    def submit_json(self, path: Union[str, os.PathLike], obj: Any, **kwargs: Any) -> None:
        """异步原子写 JSON。"""
        self.submit(lambda: atomic_write_json(path, obj, **kwargs))

    def submit_text(self, path: Union[str, os.PathLike], text: str,
                    encoding: str = "utf-8") -> None:
        """异步原子写文本。"""
        self.submit(lambda: atomic_write_text(path, text, encoding=encoding))

    def submit_append_line(self, path: Union[str, os.PathLike], line: str) -> None:
        """异步追加一行。"""
        self.submit(lambda: atomic_append_line(path, line))

    # ------------------------------------------------------------ 同步
    def flush(self, timeout: float = 30.0) -> bool:
        """阻塞直到当前已提交任务全部落盘。返回是否在超时前完成。"""
        with self._drained:
            return self._drained.wait_for(lambda: self._outstanding == 0, timeout)

    def close(self, timeout: float = 30.0) -> None:
        """排空队列并停止写线程。可重复调用。"""
        with self._drained:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is not None and thread.is_alive():
            self._queue.put(_SENTINEL)
            thread.join(timeout)
        self.flush(timeout)

    def __enter__(self) -> "WriteQueue":
        with self._drained:
            self._ensure_thread_locked()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------ 属性
    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def is_idle(self) -> bool:
        with self._drained:
            return self._outstanding == 0

    @property
    def errors(self) -> list:
        """消费过程中捕获的异常（不中断写线程）。"""
        return list(self._errors)

    # ------------------------------------------------------------ 内部
    def _ensure_thread_locked(self) -> None:
        """启动写线程。调用方必须持有 self._drained。"""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name=self._name, daemon=True
            )
            self._thread.start()

    def _finish_one(self) -> None:
        with self._drained:
            self._outstanding -= 1
            if self._outstanding <= 0:
                self._drained.notify_all()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            is_sentinel = task is _SENTINEL
            if is_sentinel:
                self._queue.task_done()
                return
            try:
                task()
            except Exception as exc:  # 单个任务失败不能拖垮写线程
                self._errors.append(exc)
            finally:
                self._queue.task_done()
                self._finish_one()


# 全局默认队列（多数调用方只需要这一个）
_default: Optional[WriteQueue] = None
_default_lock = threading.Lock()


def get_default_write_queue() -> WriteQueue:
    """进程级共享写队列（单例）。"""
    global _default
    with _default_lock:
        if _default is None:
            _default = WriteQueue()
            # 退出前排空，避免最后几条状态写丢失（atexit 晚于模块清理，
            # 这里保持函数引用以免被垃圾回收）
            atexit.register(_flush_and_close_default)
        return _default


def _flush_and_close_default() -> None:
    global _default
    with _default_lock:
        dq, _default = _default, None
    if dq is not None:
        try:
            dq.close()
        except Exception:
            pass
