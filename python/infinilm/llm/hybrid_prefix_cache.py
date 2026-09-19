"""Byte-bounded, exact-prompt snapshots for single-rank greedy hybrid MTP.

Snapshots own their device storage. Restoration writes into request-owned pages
and recurrent rows; no mutable state or partially filled KV page is shared.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass

import infinicore


@dataclass
class HybridPrefixSnapshot:
    kv: list
    states: list
    hidden: object
    proposal: object
    pending: int
    nbytes: int


class HybridPrefixCache:
    def __init__(self, engine, budget_bytes, block_size):
        self.engine = engine
        self.budget_bytes = budget_bytes
        self.block_size = block_size
        self.entries = OrderedDict()
        self.used_bytes = 0
        self.hits = self.misses = self.evictions = self.skipped = 0
        self.generation = self._generation()

    def _generation(self):
        return getattr(self.engine, "cache_generation", 0)

    def clear(self):
        self.entries.clear()
        self.used_bytes = 0
        self.generation = self._generation()

    def _check_generation(self):
        if self.generation != self._generation():
            self.clear()

    def _storage(self):
        kv = [t for t in self.engine.get_kv_cache()[0] if t._underlying is not None]
        states = [
            infinicore.Tensor(t)
            for group in self.engine.get_hybrid_states()[0]
            for t in group
            if t is not None
        ]
        return kv, states

    @staticmethod
    def _bytes(tensor):
        return (
            math.prod(tensor.shape)
            * infinicore.utils.to_torch_dtype(tensor.dtype).itemsize
        )

    @staticmethod
    def _clone(tensor):
        # InfiniCore-owned buffers keep this copy independent of foreign allocator
        # lifetimes and of the compiler's reusable graph output buffers.
        result = infinicore.empty(
            tensor.shape, dtype=tensor.dtype, device=tensor.device
        )
        result.copy_(tensor)
        return result

    def restore(self, request):
        self._check_generation()
        key = tuple(request.prompt_token_ids)
        saved = self.entries.get(key)
        if saved is None:
            self.misses += 1
            return None
        kv, states = self._storage()
        # Metadata construction can leave the calling thread on CPU. In-place
        # InfiniCore ops use that thread's handle/stream, unlike `empty()`, which
        # selects the tensor's device. Select it explicitly before restoration.
        infinicore.set_device(saved.hidden.device)
        for dst, pages in zip(kv, saved.kv):
            for block, src in zip(request.block_table, pages):
                dst.narrow(1, block, 1).copy_(src)
        for dst, src in zip(states, saved.states):
            dst.narrow(0, request.mamba_cache_index, 1).copy_(src)
        infinicore.sync_device()
        self.entries.move_to_end(key)
        self.hits += 1
        return saved

    def save(self, request, pending):
        self._check_generation()
        key = tuple(request.prompt_token_ids)
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        state = request.mtp_state
        if state is None:
            return
        kv, states = self._storage()
        pages = (request.get_prompt_length() + self.block_size - 1) // self.block_size
        sources = [
            [t.narrow(1, block, 1) for block in request.block_table[:pages]] for t in kv
        ]
        state_sources = [t.narrow(0, request.mamba_cache_index, 1) for t in states]
        tensors = (
            [x for layer in sources for x in layer]
            + state_sources
            + [state.draft_hidden]
        )
        if isinstance(state.draft_token, infinicore.Tensor):
            tensors.append(state.draft_token)
        nbytes = sum(self._bytes(t) for t in tensors)
        # Evict before allocating the new snapshot, so the live snapshots never
        # temporarily exceed the configured storage budget.
        if nbytes > self.budget_bytes:
            self.skipped += 1
            return
        while self.used_bytes + nbytes > self.budget_bytes:
            _, old = self.entries.popitem(last=False)
            self.used_bytes -= old.nbytes
            self.evictions += 1
            del old
        snapshot = HybridPrefixSnapshot(
            [[self._clone(t) for t in layer] for layer in sources],
            [self._clone(t) for t in state_sources],
            self._clone(state.draft_hidden),
            self._clone(state.draft_token)
            if isinstance(state.draft_token, infinicore.Tensor)
            else state.draft_token,
            pending,
            nbytes,
        )
        infinicore.sync_device()
        self.entries[key] = snapshot
        self.used_bytes += nbytes
