"""
KV Cache Manager - Paged Attention block-based cache allocation and management.
"""

from collections import deque
from typing import List, Dict, Set
import xxhash
import numpy as np


class Block:
    """KV Cache Block with reference counting and hash-based reuse support."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: List[int] = []

    def __repr__(self) -> str:
        return f"Block(id={self.block_id}, ref={self.ref_count}, hash={self.hash})"

    def update(self, hash_value: int, token_ids: List[int]) -> None:
        self.hash = hash_value
        self.token_ids = token_ids.copy()

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def free(self) -> None:
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Manages Paged KV Cache allocation with prefix caching support.

    Features:
    - Block allocation/deallocation with reference counting
    - Hash-based prefix caching for token sequence reuse
    - Slot mapping generation for physical-to-logical position mapping
    """

    @classmethod
    def compute_hash(
        cls,
        token_ids: List[int],
        prefix_hash: int = -1,
        mm_data_identifiers: List[str] = None,
    ) -> int:
        """Compute hash for token sequence with optional prefix chaining."""
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        if mm_data_identifiers is not None:
            for identifier in mm_data_identifiers:
                h.update(identifier.encode("utf-8"))
        return h.intdigest()

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0 and block_size > 0, (
            "num_blocks and block_size must be positive"
        )
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()
        self.pending_block_ids: Set[int] = set()

    def __repr__(self):
        return (
            f"BlockManager(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"free={len(self.free_block_ids)}, used={len(self.used_block_ids)})"
        )

    # Private low-level operations

    def _allocate_partial_block(self) -> Block:
        """Pop the first free block and add it to used blocks as a partial block."""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count not zero"

        block.reset()
        self.used_block_ids.add(block_id)
        return block

    def _allocate_full_block(self) -> Block:
        """Pop the first free block and add it to pending blocks as a full block."""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count not zero"

        block.reset()
        self.pending_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        """Deallocate a block and return it to free list."""
        block = self.blocks[block_id]
        assert block.ref_count == 0, (
            f"Block {block_id} ref_count not zero, cannot deallocate"
        )

        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        block.free()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _commit_pending_blocks(self) -> None:
        """Commit pending prefill blocks into used_block_ids and register their hashes."""
        for block_id in self.pending_block_ids:
            self.used_block_ids.add(block_id)
            block = self.blocks[block_id]
            if block.hash != -1:
                self.hash_to_block_id[block.hash] = block_id
        self.pending_block_ids.clear()

    # Read-only state queries

    def can_allocate(self, num_required_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_required_blocks

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def get_total_usable_blocks(self) -> int:
        freeable_used_blocks = sum(
            1 for bid in self.used_block_ids if self.blocks[bid].ref_count == 0
        )
        return len(self.free_block_ids) + freeable_used_blocks

    # Core public operations

    def get_computed_blocks(
        self,
        token_ids: List[int],
        mm_token_index_mappings: List[dict] = None,
    ) -> tuple[List[int], int, List[dict]]:
        """Find locally cached prefix blocks for the given token sequence.

        The last token is never matched, as it must be recomputed to obtain logits.

        Args:
            token_ids: Input token sequence.
            mm_token_index_mappings: List of multimodal token index mappings.
        Returns:
            A tuple of (cached_block_table, num_local_cached_tokens, blocks_blueprint):
            - cached_block_table: List of matched block IDs (each with ref_count incremented).
            - num_local_cached_tokens: Number of cached tokens (always a multiple of block_size).
            - blocks_blueprint: Per-block cached block id and precomputed prefix hash.
        """
        num_tokens = len(token_ids)
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        num_full_blocks = num_tokens // self.block_size
        remain_tokens = num_tokens % self.block_size
        mm_token_index_mappings = mm_token_index_mappings or []
        num_mm_inputs = len(mm_token_index_mappings)

        # Variables
        cached_block_table = []
        prefix_hash = -1
        cache_miss = False
        mm_start_counter = 0
        mm_caching_queue = deque()
        blocks_blueprint = []  # [{"prefix_hash": int or -1 if not a full block, "block_id": int or -1 if not cached}, ...]
        max_blocks_to_reuse = num_full_blocks

        for block_idx in range(num_blocks):
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, num_tokens)
            block_tokens = token_ids[start_idx:end_idx]

            # Process multimodal token index mappings for this block
            mm_data_identifiers = []
            while (
                mm_start_counter < num_mm_inputs
                and mm_token_index_mappings[mm_start_counter]["start_index"] < end_idx
                and mm_token_index_mappings[mm_start_counter]["start_index"]
                >= start_idx
            ):
                # for all mm_data whose start_index is within this block's token range, add its identifier to the list
                mm_data_identifiers.append(
                    mm_token_index_mappings[mm_start_counter]["identifier"]
                )
                mm_caching_queue.append(mm_start_counter)
                mm_start_counter += 1

            prefix_hash = (
                self.compute_hash(block_tokens, prefix_hash, mm_data_identifiers)
                if len(block_tokens) == self.block_size
                else -1
            )

            # Try to reuse existing block if no previous cache miss yet
            cached_block_id = (
                self.hash_to_block_id.get(prefix_hash, -1) if not cache_miss else -1
            )
            if (
                cached_block_id != -1
                and self.blocks[cached_block_id].token_ids != block_tokens
            ):
                cached_block_id = -1
            if end_idx == num_tokens and remain_tokens == 0:
                # Spicial case, when the last block is fully packed, we cannot reuse it because we need to leave at least one uncached token for forward
                cached_block_id = -1

            # Deal with the first cache miss
            if not cache_miss and cached_block_id == -1:
                max_blocks_to_reuse = min(max_blocks_to_reuse, block_idx)
                cache_miss = True

            if not cache_miss:
                # pop fully cached mm_data
                while (
                    mm_caching_queue
                    and mm_token_index_mappings[mm_caching_queue[0]]["end_index"]
                    < end_idx
                ):
                    mm_caching_queue.popleft()

            blocks_blueprint.append(
                {"prefix_hash": prefix_hash, "block_id": cached_block_id}
            )

        # If there is one incomplete mm_data, tailing blocks need to fall back until all included mm_data are complete
        if mm_caching_queue:
            incomplete_mm = mm_token_index_mappings[mm_caching_queue.popleft()]
            incomplete_mm_start = incomplete_mm[
                "start_index"
            ]  # Fall back until this index is no longer included in the block
            max_blocks_to_reuse = min(
                max_blocks_to_reuse, incomplete_mm_start // self.block_size
            )

        num_local_cached_tokens = max_blocks_to_reuse * self.block_size

        for block_id in range(max_blocks_to_reuse):
            block = self.blocks[blocks_blueprint[block_id]["block_id"]]
            block.ref_count += 1
            cached_block_table.append(block.block_id)

        return cached_block_table, num_local_cached_tokens, blocks_blueprint

    def allocate_slots(
        self,
        token_ids: List[int],
        num_new_tokens: int,
        num_computed_tokens: int = 0,
        cached_block_table: List[int] = None,
        blocks_blueprint: List[dict] = None,
        delay_cache_blocks: bool = False,
    ) -> tuple[List[int], List[int]] | None:
        """Allocate KV cache slots for a request (PD-disaggregation aware).

        Note: Requires that the underlying attention kernel writes KV cache before
        reading it (write-before-read ordering).

        Args:
            token_ids: Complete token sequence for the request.
            num_new_tokens: Number of tokens to compute in this step.
            num_computed_tokens: Total number of tokens already computed across local and remote workers.
            cached_block_table: Already-matched local block IDs.
            blocks_blueprint: Per-block precomputed prefix hashes from get_computed_blocks.
            delay_cache_blocks: When True (async PD transfer in progress), allocate
                blocks but defer hash registration until transfer completes.

        Returns:
            A tuple of (block_table, slot_mapping), or None if blocks are insufficient.
            - block_table: Full block list.
            - slot_mapping: Physical slot IDs for the tokens that need to be computed.
        """
        if cached_block_table is None:
            cached_block_table = []
        block_table = list(cached_block_table)
        slot_mapping = []

        total_tokens = num_computed_tokens + num_new_tokens

        num_blocks_needed = (
            total_tokens + self.block_size - 1
        ) // self.block_size - len(cached_block_table)

        if not self.can_allocate(num_blocks_needed):
            if not self.try_free_blocks(num_blocks_needed):
                return None

        start_block_idx = len(cached_block_table)
        total_blocks = (total_tokens + self.block_size - 1) // self.block_size
        prefix_hash = (
            self.blocks[cached_block_table[-1]].hash if cached_block_table else -1
        )

        for block_idx in range(start_block_idx, total_blocks):
            start_tok = block_idx * self.block_size
            end_tok = min(start_tok + self.block_size, len(token_ids))
            block_tokens = token_ids[start_tok:end_tok]
            is_full_block = len(block_tokens) == self.block_size

            if not self.free_block_ids:
                return None

            if is_full_block:
                block_hash = -1
                if blocks_blueprint is not None and block_idx < len(blocks_blueprint):
                    block_hash = blocks_blueprint[block_idx]["prefix_hash"]
                if block_hash == -1:
                    block_hash = self.compute_hash(block_tokens, prefix_hash)
                prefix_hash = block_hash
                block = self._allocate_full_block()
                block.update(block_hash, block_tokens)
            else:
                block = self._allocate_partial_block()

            block_table.append(block.block_id)

        for tok_idx in range(num_computed_tokens, total_tokens):
            blk_idx = tok_idx // self.block_size
            blk_offset = tok_idx % self.block_size
            slot_mapping.append(block_table[blk_idx] * self.block_size + blk_offset)

        if delay_cache_blocks:
            for block_id in list(self.pending_block_ids):
                self.used_block_ids.add(block_id)
            self.pending_block_ids.clear()
        else:
            self._commit_pending_blocks()

        return block_table, slot_mapping

    def append_slot(
        self, block_table: List[int], num_tokens: int, total_token_ids: List[int] = None
    ) -> tuple[List[int], int]:
        """Append slot for decode phase (generate one new token).

        Args:
            block_table: Current block_table
            num_tokens: Current total token count (including newly generated token)
            total_token_ids: All token sequence (for updating block hash)

        Returns:
            Tuple of (block_table, slot_id)
        """
        assert len(block_table) > 0, "block_table cannot be empty"
        assert num_tokens > 0, "num_tokens must be greater than 0"

        if num_tokens % self.block_size == 1:
            # Previous block is full, update its hash for future prefix caching
            last_block_id = block_table[-1]
            last_block = self.blocks[last_block_id]

            # Only update if block's token_ids is empty (avoid duplicate updates)
            if len(last_block.token_ids) == 0:
                block_start_idx = num_tokens - self.block_size - 1
                block_end_idx = num_tokens - 1
                block_tokens = total_token_ids[block_start_idx:block_end_idx]

                # Compute prefix_hash using previous block's hash if available
                if len(block_table) > 1:
                    prev_block = self.blocks[block_table[-2]]
                    prefix_hash = prev_block.hash
                else:
                    prefix_hash = -1

                current_hash = self.compute_hash(block_tokens, prefix_hash)
                last_block.update(current_hash, block_tokens)
                self.hash_to_block_id[current_hash] = last_block_id

            # Need new block
            if not self.free_block_ids:
                if not self.try_free_blocks(1):
                    raise RuntimeError("No available cache blocks")
            new_block = self._allocate_partial_block()
            block_table.append(new_block.block_id)

        # Calculate slot
        last_block_id = block_table[-1]
        offset = (num_tokens - 1) % self.block_size
        slot_id = last_block_id * self.block_size + offset

        return block_table, slot_id

    # Reference management

    def free_blocks(self, block_table: List[int]):
        """Decrease reference count for all blocks. Blocks with ref_count=0 are not
        immediately freed to allow reuse."""
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            assert block.ref_count > 0, "block ref_count must be greater than 0"
            block.ref_count -= 1

    def try_free_blocks(self, num_required: int) -> bool:
        """Try to free blocks with ref_count=0."""
        to_free = [
            bid for bid in self.used_block_ids if self.blocks[bid].ref_count == 0
        ]

        for block_id in to_free:
            self._deallocate_block(block_id)
            if self.can_allocate(num_required):
                return True

        return self.can_allocate(num_required)

    # PD-disaggregation specific

    def update_blocks_hash(self, block_table: List[int], num_local_cached_tokens: int):
        """Register hashes for blocks beyond the locally cached prefix into the lookup table.

        Called on the decode node after receiving KV data from the prefill node,
        so that subsequent requests can hit these blocks via prefix caching.
        Only full blocks (with a valid hash) are registered; partial blocks are skipped.

        Args:
            block_table: Block IDs for the current request.
            num_local_cached_tokens: Number of locally cached tokens (must be a multiple of
                block_size).
        """
        assert num_local_cached_tokens % self.block_size == 0, (
            "num_local_cached_tokens must be multiple of block_size"
        )
        for idx in range(num_local_cached_tokens // self.block_size, len(block_table)):
            block_id = block_table[idx]
            block = self.blocks[block_id]
            if block.hash != -1:
                self.hash_to_block_id[block.hash] = block_id

    def update_blocks_slot(
        self, block_table: List[int], num_computed_tokens: int, total_tokens: int
    ) -> List[int]:
        """Build the slot mapping for tokens that still need to be computed.

        Used on the decode node after a partial KV transfer failure to reconstruct
        the slot mapping covering [num_computed_tokens, total_tokens).

        Args:
            block_table: Block IDs for the current request.
            num_computed_tokens: Number of tokens already computed (may not be
                a multiple of block_size).
            total_tokens: Total token count for this request.

        Returns:
            Slot IDs for the range [num_computed_tokens, total_tokens).
        """
        bs = self.block_size
        new_slot_mapping = []

        start_block = num_computed_tokens // bs
        start_offset = num_computed_tokens % bs

        last_token_idx = total_tokens - 1
        end_block = last_token_idx // bs
        end_offset = last_token_idx % bs + 1

        if start_block == end_block:
            block_id = block_table[start_block]
            base = block_id * bs
            new_slot_mapping.extend(range(base + start_offset, base + end_offset))
            return new_slot_mapping

        block_id = block_table[start_block]
        base = block_id * bs
        new_slot_mapping.extend(range(base + start_offset, base + bs))

        for idx in range(start_block + 1, end_block):
            block_id = block_table[idx]
            base = block_id * bs
            new_slot_mapping.extend(range(base, base + bs))

        block_id = block_table[end_block]
        base = block_id * bs
        new_slot_mapping.extend(range(base, base + end_offset))

        return new_slot_mapping
