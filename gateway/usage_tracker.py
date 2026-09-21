import sqlite3
from datetime import datetime
from typing import Dict, List
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from runtime_paths import DB_DIR


class UsageTracker:
    """Track token usage per gateway key."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_DIR / "usage.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gateway_key TEXT,
                user_id TEXT,
                model TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                upstream_provider TEXT,
                response_time_ms INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_user_model
            ON usage_log(user_id, model, created_at)
        """)
        conn.commit()
        conn.close()

    def record(self, gateway_key: str, user_id: str, model: str,
               usage: Dict, provider: str, response_time_ms: int = 0):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO usage_log (gateway_key, user_id, model, prompt_tokens, "
            "completion_tokens, total_tokens, upstream_provider, response_time_ms) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (gateway_key, user_id, model,
             usage.get("prompt_tokens", 0),
             usage.get("completion_tokens", 0),
             usage.get("total_tokens", 0),
             provider, response_time_ms)
        )
        conn.commit()
        conn.close()

    def query_by_user(self, user_id: str, days: int = 30) -> List:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT model, SUM(total_tokens), COUNT(*) FROM usage_log "
            "WHERE user_id=? AND created_at > datetime('now', ?) "
            "GROUP BY model",
            (user_id, f"-{days} days")
        )
        results = cursor.fetchall()
        conn.close()
        return results

    def query_total_usage(self, gateway_key: str) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT SUM(total_tokens), SUM(prompt_tokens), SUM(completion_tokens) "
            "FROM usage_log WHERE gateway_key=?", (gateway_key,)
        )
        row = cursor.fetchone()
        conn.close()
        return {
            "total_tokens": row[0] or 0,
            "prompt_tokens": row[1] or 0,
            "completion_tokens": row[2] or 0,
        }

    def query_aggregated(self, days: int = 30) -> List:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT upstream_provider, model, SUM(total_tokens), COUNT(*) "
            "FROM usage_log WHERE created_at > datetime('now', ?) "
            "GROUP BY upstream_provider, model",
            (f"-{days} days",)
        )
        results = cursor.fetchall()
        conn.close()
        return results