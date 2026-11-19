"""Explicit configuration and calibration compatibility for replica routing."""

import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit


def positive(value, name, integer=False):
    """Reject booleans, nonfinite numbers, and invalid capacity limits."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"`{name}` must be numeric.")
    if not math.isfinite(value) or value <= 0 or (integer and int(value) != value):
        raise ValueError(f"`{name}` must be positive and finite.")
    return int(value) if integer else float(value)


class GatewayConfig:
    """Load an opt-in gateway without loading a tokenizer or model."""

    def __init__(self, filename):
        path = Path(filename)
        data = json.loads(path.read_text())
        if data.get("enabled") is not True or data.get("schema_version") != 1:
            raise ValueError("Set `schema_version=1` and `enabled=true` explicitly.")
        self.host = data.get("host", "127.0.0.1")
        self.port = positive(data.get("port", 8080), "port", True)
        self.page_size = positive(data["page_size"], "page_size", True)
        self.kv_tokens = positive(data["kv_tokens"], "kv_tokens", True)
        if self.kv_tokens % self.page_size:
            raise ValueError("`kv_tokens` must be page aligned.")
        self.max_inflight = positive(data["max_inflight"], "max_inflight", True)
        self.skew = positive(data["skew"], "skew")
        if self.skew < 1:
            raise ValueError("`skew` must be at least one.")
        self.virtual_nodes = positive(data["virtual_nodes"], "virtual_nodes", True)
        self.cache_ttl_s = positive(data["cache_ttl_s"], "cache_ttl_s")
        self.queue_timeout_s = positive(data["queue_timeout_s"], "queue_timeout_s")
        self.request_timeout_s = positive(data["request_timeout_s"], "request_timeout_s")
        self.warm_wait_ms = data["warm_wait_ms"]
        if isinstance(self.warm_wait_ms, bool) or not isinstance(self.warm_wait_ms, (int, float)):
            raise ValueError("`warm_wait_ms` must be numeric.")
        if not math.isfinite(self.warm_wait_ms) or not 0 <= self.warm_wait_ms <= 1000:
            raise ValueError("`warm_wait_ms` must be within [0, 1000].")
        self.prefix_tokens = data.get("prefix_tokens")
        if self.prefix_tokens is None:
            self.prefix_tokens = 0
        if type(self.prefix_tokens) is not int or self.prefix_tokens < 0 or self.prefix_tokens % self.page_size:
            raise ValueError("`prefix_tokens` must be a nonnegative page-aligned integer.")
        self.backends = data["backends"]
        if not isinstance(self.backends, list) or not self.backends:
            raise ValueError("Configure at least one full-model backend.")
        self.urls, self.keys = [], []
        for backend in self.backends:
            url = backend["url"].rstrip("/")
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
                raise ValueError("Backend URLs must be HTTP origins without credentials or paths.")
            key = os.environ.get(backend["key_env"], "")
            if not key:
                raise ValueError("A backend authorization environment variable is missing.")
            self.urls.append(url)
            self.keys.append(key)
        if len(set(self.urls)) != len(self.urls):
            raise ValueError("Backend URLs must be unique.")
        self.cache_policy = data["cache_policy"]
        if self.cache_policy != "prefix_lru":
            raise ValueError("This V3 entry requires explicit `prefix_lru` backends.")
        cost_path = Path(data["cost_file"])
        if not cost_path.is_absolute():
            cost_path = path.parent / cost_path
        cost = json.loads(cost_path.read_text())
        if cost.get("schema_version") != 1:
            raise ValueError("Unsupported cost file schema.")
        self.compatibility = cost["compatibility"]
        for field in ("identity", "page_size", "kv_tokens", "cache_policy", "max_inflight"):
            if field not in self.compatibility:
                raise ValueError(f"Cost compatibility requires `{field}`.")
        self.phase_model = cost["phase_model"]
        self.state_lock = Path(data["state_lock"])
        if not self.state_lock.is_absolute():
            self.state_lock = path.parent / self.state_lock

    def validate_contract(self, contract):
        """Fail closed when calibration and a running backend disagree."""
        if contract.get("schema_version") != 1:
            raise ValueError("Unsupported backend contract schema.")
        for field, expected in self.compatibility.items():
            if contract.get(field) != expected:
                raise ValueError(f"Backend differs from cost compatibility field `{field}`.")
        for field in ("page_size", "kv_tokens", "cache_policy"):
            if contract.get(field) != getattr(self, field):
                raise ValueError(f"Backend differs from gateway field `{field}`.")
        if self.compatibility["max_inflight"] != self.max_inflight:
            raise ValueError("Cost calibration and gateway concurrency disagree.")
        for field in ("backend_instance_id", "token_digest"):
            if not isinstance(contract.get(field), str) or not contract[field]:
                raise ValueError(f"Backend omitted `{field}`.")
        for field in ("prompt_tokens", "output_tokens"):
            positive(contract.get(field), field, True)
        shared = contract.get("shared_tokens")
        if type(shared) is not int or shared < 0 or shared >= contract["prompt_tokens"] or shared % self.page_size:
            raise ValueError("Backend returned an invalid prefix contract.")
        if shared and (shared != self.prefix_tokens or not contract.get("prefix_digest")):
            raise ValueError("Backend returned an unexpected prefix contract.")
