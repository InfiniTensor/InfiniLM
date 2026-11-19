"""Opt-in CPU contracts and observations for complete model replicas.

The idle-page policy in `cache_manager.py` adapts the historical V3
`cache_policy.py` algorithm to the current allocator. No old allocator is
installed, and published prefix ownership remains with the current scheduler.
"""

import asyncio
import hashlib
import inspect
import json
import math
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

from infinilm.llm.prefix_cache import EMPTY_BLOCK_HASH, hash_block_tokens
from infinilm.processors.basic_llm_processor import BasicLLMProcessor

_RUNTIME_REVISION = "ec6614ea32d778ce819aaa5c57e8ae3c7341a580"
_ALLOWED_FIELDS = {
    "model", "messages", "prompt", "stream", "stream_options",
    "sampling_params", "temperature", "top_p", "top_k", "max_tokens",
    "max_completion_tokens", "max_new_tokens", "stop",
    "add_generation_prompt", "chat_template_kwargs",
}


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


class BackendContractError(ValueError):
    """Reject a stale or unsupported contract before generation is submitted."""

    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


class PrefixObservations:
    """Publish owner-thread CPU snapshots without pinning or reading tensors."""

    def __init__(self, instance_id, max_entries=256):
        self.instance_id = instance_id
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._known = OrderedDict()
        self._epoch = 0
        self._snapshot = None

    def register(self, digest, hashes):
        with self._lock:
            self._known.pop(digest, None)
            self._known[digest] = tuple(hashes)
            while len(self._known) > self.max_entries:
                self._known.popitem(last=False)

    def publish(self, manager):
        """Read only published metadata on the scheduler's resource owner."""
        with self._lock:
            prefixes = {}
            for digest, hashes in self._known.items():
                resident = 0
                for block_hash in hashes:
                    block_ids = manager.hash_to_block_ids.get(block_hash, ())
                    if not any(
                        block_id in manager.used_block_ids
                        and manager.blocks[block_id].hash == block_hash
                        for block_id in block_ids
                    ):
                        break
                    resident += manager.block_size
                prefixes[digest] = resident
            self._epoch += 1
            self._snapshot = {
                "backend_instance_id": self.instance_id,
                "epoch": self._epoch,
                "prefixes": prefixes,
                "observed_monotonic": time.monotonic(),
            }

    def snapshot(self):
        with self._lock:
            if self._snapshot is None:
                return {
                    "backend_instance_id": self.instance_id,
                    "epoch": 0,
                    "prefixes": {},
                    "age_ms": None,
                }
            value = dict(self._snapshot)
            value["prefixes"] = dict(value["prefixes"])
            value["age_ms"] = max(
                0.0, (time.monotonic() - value.pop("observed_monotonic")) * 1000
            )
            return value


class BoundedHashBackend:
    """Bind the loaded backend tokenizer, capacity, and cache observations."""

    def __init__(self, server, key_env, cache_policy):
        self.server = server
        secret = os.environ.get(key_env, "")
        if not secret:
            raise ValueError("The bounded-hash backend secret is missing.")
        self._authorization = "Bearer " + secret
        if (
            type(server.bounded_hash_max_inflight) is not int
            or not 1 <= server.bounded_hash_max_inflight <= server.max_batch_size
        ):
            raise ValueError("Routing inflight limit must fit `max_batch_size`.")
        if not isinstance(server.bounded_hash_hardware_id, str) or not server.bounded_hash_hardware_id.strip():
            raise ValueError("An explicit routing hardware identity is required.")
        self.instance_id = uuid.uuid4().hex
        self.observations = PrefixObservations(self.instance_id)
        engine = server.engine.engine
        scheduler = engine.scheduler
        model = engine.model_runner.model_engine
        processor = engine.processor
        if (
            server.cache_type != "paged"
            or server.tensor_parallel_size != 1
            or server.pipeline_parallel_size != 1
            or server.moe_ep_backend != "disabled"
            or server.moe_ep_size != 1
            or server.kv_transfer_config is not None
            or server.use_mla
            or server.chunked_prefill_size
            or server.tool_call_parser != "none"
            or server.skip_load
            or not server.enable_prefix_caching
            or model.model_type not in {"qwen3", "llama"}
            or type(processor).__call__ is not BasicLLMProcessor.__call__
            or engine.model_runner.speculative_runner is not None
            or scheduler.has_mamba_cache
            or scheduler._nsa_owner_clock is not None
        ):
            raise ValueError(
                "Bounded-hash routing supports only TP1 dense Qwen3/Llama "
                "text replicas with paged prefix caching and whole prefills."
            )
        self.max_context_tokens = int(model.hf_config.get(
            "max_position_embeddings", server.max_cache_len
        ))
        self.cache_policy = cache_policy
        scheduler.cache_manager.configure_routing_cache_policy(cache_policy)
        scheduler.bounded_hash_observer = self.observations
        self.observations.publish(scheduler.cache_manager)
        model_dir = Path(server.model_path)
        tokenizer_files = (
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "vocab.json",
            "merges.txt", "chat_template.jinja",
        )
        token_hashes = {
            name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
            for name in tokenizer_files if (model_dir / name).is_file()
        }
        if not token_hashes:
            raise ValueError("Tokenizer artifacts are required for routing identity.")
        package = Path(__file__).resolve().parents[1]
        sources = ("llm/cache_manager.py", "llm/prefix_cache.py",
                   "processors/basic_llm_processor.py")
        self.identity = {
            "model_config_sha256": hashlib.sha256(
                (model_dir / "config.json").read_bytes()
            ).hexdigest(),
            "tokenizer_sha256": _digest(token_hashes),
            "dtype": server.dtype,
            "device": server.device.split(":", 1)[0],
            "hardware_id": server.bounded_hash_hardware_id,
            "tensor_parallel_size": server.tensor_parallel_size,
            "runtime_revision": _RUNTIME_REVISION,
            "execution_config": {
                "attn_backend": server.attn_backend,
                "enable_graph": server.enable_graph,
                "pre_transpose": server.pre_transpose,
                "max_batch_size": server.max_batch_size,
                "ignore_eos": server.ignore_eos,
                "weight_load_mode": server.weight_load_mode,
                "chunked_prefill_size": server.chunked_prefill_size,
                "cache_type": server.cache_type,
                "enable_prefix_caching": server.enable_prefix_caching,
                "use_mla": server.use_mla,
                "max_cache_len": server.max_cache_len,
                "max_tokens": server.max_tokens,
                "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            },
            "processor": type(processor).__module__ + "." + type(processor).__name__,
            "processor_sha256": hashlib.sha256(
                Path(inspect.getfile(type(processor))).read_bytes()
            ).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                for name in sources
            },
        }

    def authorize(self, headers):
        if not secrets.compare_digest(
            headers.get("authorization", ""), self._authorization
        ):
            raise BackendContractError("Backend authentication failed.", 401)

    def normalize(self, data):
        if not isinstance(data, dict) or set(data) - _ALLOWED_FIELDS:
            raise BackendContractError("Unsupported bounded-hash request fields.", 400)
        data = dict(data)
        messages = data.get("messages")
        if not messages and isinstance(data.get("prompt"), str):
            messages = [{"role": "user", "content": data["prompt"]}]
            data["messages"] = messages
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") not in {"user", "assistant", "system"}
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise BackendContractError("Only plain text chat messages are supported.", 400)
        if "model" in data and data["model"] != self.server.model_id:
            raise BackendContractError("The requested model is not served here.", 400)
        if not isinstance(data.get("add_generation_prompt", True), bool):
            raise BackendContractError("Invalid `add_generation_prompt`.", 400)
        if not isinstance(data.get("chat_template_kwargs") or {}, dict):
            raise BackendContractError("Invalid `chat_template_kwargs`.", 400)
        if set(data.get("chat_template_kwargs") or {}) - {"enable_thinking"}:
            raise BackendContractError("Unsupported chat template arguments.", 400)
        if "enable_thinking" in (data.get("chat_template_kwargs") or {}) and not isinstance(
            data["chat_template_kwargs"]["enable_thinking"], bool
        ):
            raise BackendContractError("Invalid `enable_thinking`.", 400)
        if not isinstance(data.get("sampling_params") or {}, dict):
            raise BackendContractError("Invalid `sampling_params`.", 400)
        if set(data.get("sampling_params") or {}) - {
            "temperature", "top_p", "top_k", "max_tokens",
            "max_completion_tokens", "max_new_tokens", "stop",
        }:
            raise BackendContractError("Unsupported sampling parameters.", 400)
        if not isinstance(data.get("stream", False), bool):
            raise BackendContractError("Invalid `stream`.", 400)
        options = data.get("stream_options") or {}
        if not isinstance(options, dict) or set(options) - {"include_usage"}:
            raise BackendContractError("Unsupported stream options.", 400)
        if not isinstance(options.get("include_usage", False), bool):
            raise BackendContractError("Invalid `include_usage`.", 400)
        sampling = self.server._build_sampling_params(data)
        if (
            type(sampling.max_tokens) is not int or sampling.max_tokens <= 0
            or not math.isfinite(sampling.temperature) or sampling.temperature < 0
            or not math.isfinite(sampling.top_p) or not 0 < sampling.top_p <= 1
            or sampling.top_k < 0
            or not isinstance(sampling.stop, list)
            or any(not isinstance(stop, str) for stop in sampling.stop)
        ):
            raise BackendContractError("Invalid sampling limits.", 400)
        return data, sampling

    def contract(self, data, prefix_tokens=0, register=True):
        data, sampling = self.normalize(data)
        engine = self.server.engine.engine
        kwargs = data.get("chat_template_kwargs") or {}
        add_prompt = data.get("add_generation_prompt", True)
        prompt = engine.apply_chat_template(
            data["messages"], add_generation_prompt=add_prompt,
            chat_template_kwargs=kwargs,
        )
        processed = engine.process(
            prompt, [], [], [], chat_messages=data["messages"],
            chat_template_kwargs=kwargs, add_generation_prompt=add_prompt,
            return_tensors="np",
        )
        tokens = [int(token) for token in processed["input_ids"].flatten().tolist()]
        prefix_tokens = 0 if prefix_tokens is None else prefix_tokens
        page = self.server.block_size
        if (
            type(prefix_tokens) is not int or prefix_tokens < 0
            or prefix_tokens % page
        ):
            raise BackendContractError("Prefix must be page aligned.", 400)
        if not tokens:
            raise BackendContractError("The tokenized prompt is empty.", 400)
        if prefix_tokens >= len(tokens):
            prefix_tokens = 0
        if len(tokens) + sampling.max_tokens > self.max_context_tokens:
            raise BackendContractError("The full sequence exceeds model context.", 400)
        capacity = self.server.num_blocks * page
        if math.ceil((len(tokens) + sampling.max_tokens) / page) * page > capacity:
            raise BackendContractError("The full KV reservation exceeds capacity.", 400)
        parent = EMPTY_BLOCK_HASH
        hashes = []
        for start in range(0, prefix_tokens, page):
            parent = hash_block_tokens(tokens[start:start + page], parent)
            hashes.append(parent)
        prefix_digest = _digest(tokens[:prefix_tokens]) if prefix_tokens else None
        if hashes and register:
            self.observations.register(prefix_digest, hashes)
        token_digest = _digest({
            "token_ids": tokens, "sampling": asdict(sampling),
            "add_generation_prompt": add_prompt, "chat_template_kwargs": kwargs,
        })
        snapshot = self.observations.snapshot()
        return {
            "schema_version": 1, "backend_instance_id": self.instance_id,
            "epoch": snapshot["epoch"], "identity": self.identity,
            "prompt_tokens": len(tokens), "output_tokens": sampling.max_tokens,
            "shared_tokens": prefix_tokens, "prefix_digest": prefix_digest,
            "key": prefix_digest or token_digest, "token_digest": token_digest,
            "page_size": page, "kv_tokens": capacity,
            "cache_policy": self.cache_policy, "cache_snapshot": snapshot,
            "max_inflight": self.server.bounded_hash_max_inflight,
        }

    def validate_generation(self, data, headers):
        self.authorize(headers)
        if headers.get("x-infinilm-backend-instance") != self.instance_id:
            raise BackendContractError("Backend instance changed before generation.")
        try:
            prefix = int(headers.get("x-infinilm-prefix-tokens", "0"))
        except ValueError as error:
            raise BackendContractError("Invalid prefix header.", 400) from error
        actual = self.contract(data, prefix)
        if headers.get("x-infinilm-token-digest") != actual["token_digest"]:
            raise BackendContractError("Token contract changed before generation.")
        return self.normalize(data)[0]

    def info(self, request):
        return {
            "backend_instance_id": self.instance_id,
            "generated_tokens": request.get_num_generated_tokens(),
            "cache_policy": self.cache_policy,
            "initial_cached_tokens": getattr(
                request, "_bounded_hash_initial_cached_tokens", None
            ),
            "cache_snapshot": self.observations.snapshot(),
        }

    def usage(self, request):
        return {
            "prompt_tokens": request.get_prompt_length(),
            "completion_tokens": request.get_num_generated_tokens(),
            "total_tokens": request.get_total_length(),
            "prompt_tokens_details": {
                "cached_tokens": getattr(
                    request, "_bounded_hash_initial_cached_tokens", None
                ),
            },
        }

    async def wait_retired(self, request):
        """Wait for CPU resource retirement before acknowledging completion."""
        deadline = time.monotonic() + 5.0
        while request._paged_completion_state != "done":
            if not self.server.engine.is_healthy() or time.monotonic() >= deadline:
                raise RuntimeError("Backend resource retirement was not confirmed.")
            await asyncio.sleep(0.001)
