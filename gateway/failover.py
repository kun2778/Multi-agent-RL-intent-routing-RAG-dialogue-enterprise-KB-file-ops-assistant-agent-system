import httpx
from typing import Optional, Dict
from datetime import datetime


class FailoverHandler:
    """Automatic failover with health checking."""

    def __init__(self, key_pool, max_retries: int = 3):
        self.key_pool = key_pool
        self.max_retries = max_retries
        self._failure_counts = {}
        self._last_failure = {}

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
                        # Reset failure count on success
                        self._failure_counts[key] = 0
                        return resp
                    last_error = Exception(f"Upstream {resp.status_code}")
                    self.key_pool.mark_unhealthy(key)
                    self._failure_counts[key] = self._failure_counts.get(key, 0) + 1

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = e
                self.key_pool.mark_unhealthy(key)
                self._failure_counts[key] = self._failure_counts.get(key, 0) + 1

        raise last_error or Exception("All failover attempts exhausted")

    def is_healthy(self, key: str, threshold: int = 3) -> bool:
        """Check if a key has failed fewer than threshold times."""
        return self._failure_counts.get(key, 0) < threshold

    def get_status(self) -> Dict:
        """Get current failover status."""
        return {
            "failure_counts": self._failure_counts,
            "healthy_keys": self.key_pool.healthy_keys,
            "unhealthy_keys": [k for k, v in self._failure_counts.items() if v >= 3]
        }