"""
SQLite 注册表模块：替代 JSON 存储，支持增量扫描和百万级文件。

功能：
  - SQLite 存储文件元数据（path, size, mtime, content_hash）
  - 增量扫描：仅重新读取 mtime 或 hash 变化的文件
  - 不存储文件全文，按需读取
  - 支持批量插入/更新/删除
"""

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple


class RegistryDB:
    """SQLite 文件注册表。"""

    def __init__(self, db_path: str):
        """
        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self):
        """确保数据库和表结构存在。"""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                title TEXT,
                type TEXT,
                tags TEXT DEFAULT '[]',
                size INTEGER DEFAULT 0,
                mtime REAL DEFAULT 0,
                readable INTEGER DEFAULT 0,
                content_hash TEXT DEFAULT '',
                indexed_at REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_files_readable ON files(readable)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_files_type ON files(type)
        """)
        conn.commit()
        conn.close()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接（上下文管理器）。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_all_paths(self) -> Set[str]:
        """获取所有已注册的文件路径。"""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT path FROM files").fetchall()
            return {row["path"] for row in rows}

    def get_file(self, path: str) -> Optional[Dict]:
        """获取单个文件的元数据。"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
            if row:
                return dict(row)
            return None

    def get_files_by_path(self, paths: List[str]) -> Dict[str, Dict]:
        """批量获取文件元数据。"""
        if not paths:
            return {}
        result = {}
        with self._get_conn() as conn:
            for i in range(0, len(paths), 500):
                batch = paths[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT * FROM files WHERE path IN ({placeholders})", batch
                ).fetchall()
                for row in rows:
                    result[row["path"]] = dict(row)
        return result

    def get_all_readable(self) -> List[Dict]:
        """获取所有可读文件。"""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM files WHERE readable = 1").fetchall()
            return [dict(row) for row in rows]

    def upsert_file(self, path: str, title: str, file_type: str, size: int,
                    mtime: float, readable: bool, content_hash: str = "",
                    tags: str = "[]") -> bool:
        """插入或更新文件记录。

        Returns:
            True if inserted, False if updated (mtime unchanged).
        """
        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT mtime, content_hash FROM files WHERE path = ?", (path,)
            ).fetchone()

            if existing and existing["mtime"] == mtime and existing["content_hash"] == content_hash:
                return False  # 无变化，跳过

            conn.execute("""
                INSERT OR REPLACE INTO files
                (path, title, type, tags, size, mtime, readable, content_hash, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (path, title, file_type, tags, size, mtime, int(readable),
                  content_hash, time.time()))
            return True

    def delete_file(self, path: str) -> bool:
        """删除文件记录。"""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM files WHERE path = ?", (path,))
            return cursor.rowcount > 0

    def delete_files(self, paths: Set[str]) -> int:
        """批量删除文件记录。"""
        if not paths:
            return 0
        with self._get_conn() as conn:
            total = 0
            for i in range(0, len(paths), 500):
                batch = list(paths)[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                cursor = conn.execute(
                    f"DELETE FROM files WHERE path IN ({placeholders})", batch
                )
                total += cursor.rowcount
            return total

    def count(self) -> int:
        """获取文件总数。"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM files").fetchone()
            return row["cnt"]

    def count_readable(self) -> int:
        """获取可读文件总数。"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM files WHERE readable = 1").fetchone()
            return row["cnt"]

    def get_stats(self) -> Dict:
        """获取统计信息。"""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM files").fetchone()["cnt"]
            readable = conn.execute(
                "SELECT COUNT(*) as cnt FROM files WHERE readable = 1"
            ).fetchone()["cnt"]
            types = conn.execute(
                "SELECT type, COUNT(*) as cnt FROM files GROUP BY type ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            return {
                "total_files": total,
                "readable_files": readable,
                "types": {row["type"]: row["cnt"] for row in types},
            }

    def clear(self):
        """清空所有记录。"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM files")

    # ──────────── 向后兼容接口 ────────────

    def __len__(self) -> int:
        """支持 len(registry) 调用。"""
        return self.count()

    def keys(self):
        """支持 registry.keys() 调用。"""
        return self.get_all_paths()

    def __contains__(self, path: str) -> bool:
        """支持 path in registry 调用。"""
        return self.get_file(path) is not None

    def __getitem__(self, path: str) -> Dict:
        """支持 registry[path] 调用。"""
        doc = self.get_file(path)
        if doc is None:
            raise KeyError(path)
        return doc

    def __setitem__(self, path: str, data: Dict):
        """支持 registry[path] = data 调用。"""
        self.upsert_file(
            path=path,
            title=data.get("title", ""),
            file_type=data.get("type", "unknown"),
            size=data.get("size", 0),
            mtime=data.get("mtime", 0),
            readable=data.get("readable", False),
            content_hash=data.get("content_hash", ""),
            tags=data.get("tags", "[]"),
        )

    def pop(self, path: str, default=None):
        """支持 registry.pop(path) 调用。"""
        doc = self.get_file(path)
        if doc is None:
            return default
        self.delete_file(path)
        return doc

    def get(self, path: str, default=None):
        """支持 registry.get(path) 调用。"""
        doc = self.get_file(path)
        return doc if doc is not None else default

    def close(self):
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None


def compute_file_hash(path: str, chunk_size: int = 8192) -> str:
    """计算文件内容的 SHA256 哈希。

    大文件（>= open_read 的 mmap 阈值）直接把只读 mmap 整体喂给哈希器：
    零拷贝，且不必按 8KB 逐次系统调用；小文件仍分块流式读。
    """
    from atomic_io import open_read
    h = hashlib.sha256()
    try:
        with open_read(path) as raw:
            if isinstance(raw, (bytes, bytearray)):
                step = max(chunk_size, 1 << 20)
                view = memoryview(raw)
                for i in range(0, len(view), step):
                    h.update(view[i:i + step])
            else:
                h.update(raw)
        return h.hexdigest()
    except OSError:
        return ""
