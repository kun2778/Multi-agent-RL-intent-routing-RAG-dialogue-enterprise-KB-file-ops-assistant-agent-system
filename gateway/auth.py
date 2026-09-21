from fastapi import Request, HTTPException, Depends
from typing import Optional, Dict
import hashlib
import sqlite3
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from runtime_paths import DB_DIR


class GatewayAuth:
    """Gateway key authentication and management."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_DIR / "gateway_keys.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gateway_keys (
                key TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                active INTEGER DEFAULT 1,
                max_requests INTEGER DEFAULT -1,
                request_count INTEGER DEFAULT 0,
                rate_limit INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def create_gateway_key(self, user_id: str, expires_days: int = 90,
                           max_requests: int = -1, rate_limit: int = 0) -> str:
        raw = f"sk-gw-{user_id}-{datetime.utcnow().timestamp()}-{hashlib.sha256(user_id.encode()).hexdigest()[:16]}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:32]
        expires_at = datetime.utcnow() + timedelta(days=expires_days)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO gateway_keys VALUES (?,?,?,?,?,?,?,?)",
            (key, user_id, datetime.utcnow(), expires_at, 1, max_requests, 0, rate_limit)
        )
        conn.commit()
        conn.close()
        return key

    def validate(self, request_key: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT * FROM gateway_keys WHERE key=? AND active=1",
            (request_key,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        key, user_id, created_at, expires_at, active, max_req, req_count, rate_limit = row

        if expires_at and datetime.fromisoformat(expires_at) < datetime.utcnow():
            return None
        if max_req > 0 and req_count >= max_req:
            return None

        return {
            "key": key, "user_id": user_id,
            "expires_at": expires_at,
            "request_count": req_count, "rate_limit": rate_limit
        }

    def verify_request(self, request: Request) -> Optional[Dict]:
        """Verify a FastAPI request's API key."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing API key")
        key = auth_header[7:]
        record = self.validate(key)
        if not record:
            raise HTTPException(401, "Invalid or expired API key")
        return record