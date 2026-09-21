from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from runtime_paths import DB_DIR


# ---- Configuration ----

@dataclass
class GatewayAuthConfig:
    token_expires_days: int = 90
    active: bool = True


@dataclass
class KeyPoolConfig:
    keys: List[Dict[str, Any]] = field(default_factory=list)
    strategy: str = "weighted_round_robin"  # "weighted_round_robin", "random", "sequential"


@dataclass
class FailoverConfig:
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 30
    unhealthy_threshold: int = 3


@dataclass
class UsageTrackerConfig:
    db_path: str = None
    retention_days: int = 30

    def __post_init__(self):
        if self.db_path is None:
            self.db_path = str(DB_DIR / "usage.db")


# ---- Auth ----

class GatewayAuth:
    """Gateway key authentication and management."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_DIR / "gateway_keys.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the gateway keys database."""
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
        """Generate a new gateway key for a user."""
        raw = f"sk-gw-{user_id}-{datetime.utcnow().timestamp()}-{os.urandom(16).hex()}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:32]
        expires_at = datetime.utcnow() + timedelta(days=expires_days)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO gateway_keys VALUES (?,?,?,?,?,?,?,?)",
            (key, user_id, datetime.utcnow(), expires_at, 1,
             max_requests, 0, rate_limit)
        )
        conn.commit()
        conn.close()
        return key

    def validate(self, request_key: str) -> Optional[Dict]:
        """Validate a gateway key. Returns user info or None."""
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
            "key": key,
            "user_id": user_id,
            "expires_at": expires_at,
            "request_count": req_count,
            "rate_limit": rate_limit
        }

    def increment_usage(self, key: str):
        """Increment request count for a key."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE gateway_keys SET request_count = request_count + 1 WHERE key=?",
            (key,)
        )
        conn.commit()
        conn.close()


# ---- Key Pool ----

class KeyPool:
    """Multi-key load balancing pool."""

    def __init__(self, keys: List[Dict[str, Any]]):
        self.keys = keys
        self._expanded: List[str] = []
        for k in keys:
            weight = k.get("weight", 1)
            self._expanded.extend([k["key"]] * weight)
        self._index = 0
        self._healthy = {k["key"]: True for k in keys}

    def next_key(self) -> str:
        """Get next key using weighted round-robin."""
        healthy_keys = [k for k in self._expanded if self._healthy.get(k, True)]
        if not healthy_keys:
            return None
        key = healthy_keys[self._index % len(healthy_keys)]
        self._index += 1
        return key

    def random_key(self) -> str:
        """Get a random healthy key."""
        healthy_keys = [k for k in self._expanded if self._healthy.get(k, True)]
        import random
        return random.choice(healthy_keys) if healthy_keys else None

    def mark_unhealthy(self, key: str):
        """Mark a key as unhealthy."""
        self._healthy[key] = False

    def mark_healthy(self, key: str):
        """Mark a key as healthy."""
        self._healthy[key] = True

    def health_check(self, adapter_url: str, adapter_name: str) -> Dict[str, bool]:
        """Health check for all keys."""
        results = {}
        for key in [k["key"] for k in self.keys]:
            try:
                import requests
                resp = requests.get(f"{adapter_url}/v1/models", timeout=5,
                                   headers={"Authorization": f"Bearer {key}"})
                results[key] = resp.status_code == 200
            except Exception:
                results[key] = False
        return results


# ---- Failover ----

class FailoverHandler:
    """Automatic failover with health checking."""

    def __init__(self, key_pool: KeyPool, max_retries: int = 3):
        self.key_pool = key_pool
        self.max_retries = max_retries
        self._failure_counts: Dict[str, int] = {}

    async def call_with_failover(self, url: str, headers: Dict, body: Dict) -> httpx.Response:
        """Make request with automatic failover."""
        last_error = None

        for attempt in range(self.max_retries):
            key = self.key_pool.next_key()
            if not key:
                raise Exception("No healthy keys available")

            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        url,
                        headers={**headers, "Authorization": f"Bearer {key}"},
                        json=body
                    )
                    if resp.status_code < 500:
                        return resp
                    last_error = Exception(f"Upstream {resp.status_code}")
                    self.key_pool.mark_unhealthy(key)
                    self._failure_counts[key] = self._failure_counts.get(key, 0) + 1

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = e
                self.key_pool.mark_unhealthy(key)
                self._failure_counts[key] = self._failure_counts.get(key, 0) + 1

        raise last_error or Exception("All failover attempts exhausted")


# ---- Usage Tracker ----

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
        """Record usage from API response."""
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
        """Query usage by user."""
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
        """Get total usage for a gateway key."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT SUM(total_tokens), SUM(prompt_tokens), SUM(completion_tokens) "
            "FROM usage_log WHERE gateway_key=?",
            (gateway_key,)
        )
        row = cursor.fetchone()
        conn.close()
        return {
            "total_tokens": row[0] or 0,
            "prompt_tokens": row[1] or 0,
            "completion_tokens": row[2] or 0,
        }


# ---- FastAPI App ----

def create_gateway_app():
    """Create the FastAPI gateway application."""
    app = FastAPI(title="OpenAI Compatible Gateway", version="2.0.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize components
    auth = GatewayAuth()
    key_pool = KeyPool([])
    tracker = UsageTracker()

    @app.on_event("startup")
    async def startup():
        """Initialize gateway on startup."""
        # Load keys from config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'gateway.yaml')
        if os.path.exists(config_path):
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
                key_pool = KeyPool(config.get('key_pool', []))

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        """Authenticate all requests."""
        if request.url.path == "/health":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "Missing API key"})

        key = auth_header[7:]
        record = auth.validate(key)
        if not record:
            return JSONResponse(status_code=401, content={"error": "Invalid or expired API key"})

        request.state.user = record
        auth.increment_usage(key)
        return await call_next(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, user: dict = Depends(lambda: getattr(request, 'state', {}).get('user'))):
        """Chat completions endpoint."""
        body = await request.json()
        model = body.get("model", "gpt-4o")

        # Get adapter and route
        from providers import ProviderManager, registry
        pm = ProviderManager()
        provider_type = registry.get_route(model) or registry.default_provider
        adapter = pm.get_adapter(provider_type)

        # Get API key from Vault or env
        api_key = pm.get_api_key(provider_type)
        body["model"] = model

        # Make request with failover
        failover = FailoverHandler(key_pool, max_retries=3)
        headers = {"Content-Type": "application/json"}
        start_time = datetime.utcnow()

        try:
            if body.get("stream"):
                return StreamingResponse(
                    _stream_response(adapter, body, api_key),
                    media_type="text/event-stream"
                )
            else:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        adapter.base_url + "/v1/chat/completions",
                        headers={**headers, "Authorization": f"Bearer {api_key}"},
                        json=body
                    )
                    result = resp.json()

                    # Track usage
                    usage = result.get("usage", {})
                    elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
                    tracker.record(
                        gateway_key=user["key"],
                        user_id=user["user_id"],
                        model=model,
                        usage=usage,
                        provider=provider_type,
                        response_time_ms=int(elapsed)
                    )

                    return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": str(e)})

    @app.get("/v1/models")
    async def list_models(user: dict = Depends(lambda: getattr(request, 'state', {}).get('user'))):
        """List available models."""
        from providers import ProviderManager
        pm = ProviderManager()
        models = pm.list_available_models()
        return {"data": [{"id": name} for name in models.keys()]}

    @app.get("/usage")
    async def get_usage(user: dict = Depends(lambda: getattr(request, 'state', {}).get('user'))):
        """Get usage statistics."""
        return tracker.query_by_user(user["user_id"])

    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    return app


async def _stream_response(adapter, body: dict, api_key: str):
    """Stream response from upstream adapter."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                adapter.base_url + "/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={**body, "stream": True},
                timeout=None
            )
            async for line in resp.aiter_lines():
                yield line + "\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# Create app instance
gateway_app = create_gateway_app()