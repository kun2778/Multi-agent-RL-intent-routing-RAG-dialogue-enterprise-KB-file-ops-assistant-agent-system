import itertools
import random
from typing import Dict, List, Optional, Any


class KeyPool:
    """Multi-key load balancing pool with health checking."""

    def __init__(self, keys: List[Dict[str, Any]]):
        self.keys = keys
        self._expanded: List[str] = []
        self._healthy = {}
        self._setup_keys()

    def _setup_keys(self):
        for k in self.keys:
            weight = k.get("weight", 1)
            key_str = k.get("key", "")
            self._expanded.extend([key_str] * weight)
            self._healthy[key_str] = True
        self._index = 0
        self._cycle = itertools.cycle(self._expanded) if self._expanded else iter([])

    def next_key(self) -> Optional[str]:
        """Get next key using weighted round-robin."""
        healthy = [k for k in self._expanded if self._healthy.get(k, False)]
        if not healthy:
            return None
        key = healthy[self._index % len(healthy)]
        self._index += 1
        return key

    def random_key(self) -> Optional[str]:
        """Get a random healthy key."""
        healthy = [k for k in self._expanded if self._healthy.get(k, False)]
        return random.choice(healthy) if healthy else None

    def mark_unhealthy(self, key: str):
        self._healthy[key] = False

    def mark_healthy(self, key: str):
        self._healthy[key] = True

    def health_check(self, adapter_url: str, adapter_name: str) -> Dict[str, bool]:
        """Health check for all keys."""
        import requests
        results = {}
        for k in [k.get("key", "") for k in self.keys]:
            try:
                resp = requests.get(f"{adapter_url}/v1/models", timeout=5,
                                    headers={"Authorization": f"Bearer {k}"})
                results[k] = resp.status_code == 200
            except Exception:
                results[k] = False
        return results

    @property
    def healthy_keys(self) -> List[str]:
        return [k for k in self._expanded if self._healthy.get(k, False)]