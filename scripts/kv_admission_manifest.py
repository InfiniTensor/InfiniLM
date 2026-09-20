"""Load and validate the fixed request set used by KV admission benchmarks.

The benchmark layers intentionally share this small data format.  Keeping the
request text and token ids together makes it possible to check that the HTTP
client, the scheduler model, and the unit tests are measuring the same input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    wave: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    ordinal: int

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)


@dataclass(frozen=True)
class AdmissionManifest:
    path: Path
    model: str
    block_size: int
    num_blocks: int
    seed: int
    requests: tuple[RequestSpec, ...]
    metadata: dict[str, Any]

    def wave(self, name: str) -> tuple[RequestSpec, ...]:
        return tuple(request for request in self.requests if request.wave == name)

    @property
    def wave1(self) -> tuple[RequestSpec, ...]:
        return self.wave("wave1")

    @property
    def wave2(self) -> tuple[RequestSpec, ...]:
        return self.wave("wave2")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"`{field}` must be a non-empty string.")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{field}` must be a positive integer.")
    return value


def _parse_request(raw: Any, ordinal: int) -> RequestSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"`requests[{ordinal}]` must be an object.")

    request_id = _required_string(raw.get("request_id"), "request_id")
    wave = _required_string(raw.get("wave"), f"{request_id}.wave")
    if wave not in {"wave1", "wave2"}:
        raise ValueError(f"`{request_id}.wave` must be `wave1` or `wave2`.")
    prompt = _required_string(raw.get("prompt"), f"{request_id}.prompt")
    token_ids = raw.get("prompt_token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"`{request_id}.prompt_token_ids` must be a non-empty list.")
    if any(
        isinstance(token, bool) or not isinstance(token, int) for token in token_ids
    ):
        raise ValueError(f"`{request_id}.prompt_token_ids` must contain integers.")
    max_tokens = _positive_int(raw.get("max_tokens"), f"{request_id}.max_tokens")
    declared_length = raw.get("prompt_tokens", len(token_ids))
    if declared_length != len(token_ids):
        raise ValueError(
            f"`{request_id}.prompt_tokens={declared_length}` does not match "
            f"{len(token_ids)} token ids."
        )

    return RequestSpec(
        request_id=request_id,
        wave=wave,
        prompt=prompt,
        prompt_token_ids=tuple(token_ids),
        max_tokens=max_tokens,
        ordinal=ordinal,
    )


def load_manifest(path: str | Path) -> AdmissionManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be an object.")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported manifest `schema_version`.")

    config = payload.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("Manifest `configuration` must be an object.")
    model = _required_string(config.get("model"), "configuration.model")
    block_size = _positive_int(config.get("block_size"), "configuration.block_size")
    num_blocks = _positive_int(config.get("num_blocks"), "configuration.num_blocks")
    seed = config.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Configuration `seed` must be an integer.")

    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("Manifest `requests` must be a non-empty list.")
    requests = tuple(
        _parse_request(raw, index) for index, raw in enumerate(raw_requests)
    )
    ids = [request.request_id for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("Request `request_id` values must be unique.")
    if not any(request.wave == "wave1" for request in requests):
        raise ValueError("Manifest must contain at least one `wave1` request.")
    if not any(request.wave == "wave2" for request in requests):
        raise ValueError("Manifest must contain at least one `wave2` request.")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Manifest `metadata` must be an object.")
    return AdmissionManifest(
        path=manifest_path,
        model=model,
        block_size=block_size,
        num_blocks=num_blocks,
        seed=seed,
        requests=requests,
        metadata=metadata,
    )


def manifest_payload(
    *,
    model: str,
    block_size: int,
    num_blocks: int,
    seed: int,
    requests: Iterable[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable payload for the manifest generator."""
    return {
        "schema_version": 1,
        "configuration": {
            "model": model,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "seed": seed,
        },
        "metadata": metadata or {},
        "requests": list(requests),
    }
