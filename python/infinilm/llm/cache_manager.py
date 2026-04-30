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

    def __repr__(self) -> str:
        return f"Block(id={self.block_id}, ref={self.ref_count}, hash={self.hash})"


class BlockManager:
    """Manages Paged KV Cache allocation with prefix caching support.

    Features:
    - Block allocation/deallocation with reference counting
    - Hash-based prefix caching for token sequence reuse
    - Slot mapping generation for physical-to-logical position mapping
    """

    def __init__(self, num_blocks: int, block_size: int):
        assert (
            num_blocks > 0 and block_size > 0
        ), "num_blocks and block_size must be positive"
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()
        self.req_block_ids: Set[int] = set()

    def reset_req_blocks(self) -> None:
        """Move blocks from prefill stage to used blocks and update hash mappings."""
        for block_id in self.req_block_ids:
            self.used_block_ids.add(block_id)
            block = self.blocks[block_id]
            prefix_hash = block.hash
            self.hash_to_block_id[prefix_hash] = block_id
        self.req_block_ids.clear()

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix_hash: int = -1) -> int:
        """Compute hash for token sequence with optional prefix chaining."""
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def _allocate_partial_block(self, block_id: int) -> Block:
        """Allocate an incomplete block and add to used blocks."""
        assert block_id in self.free_block_ids, f"Block {block_id} not in free list"
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count not zero"

        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _allocate_full_block(self, block_id: int) -> Block:
        """Allocate a complete block and add to request blocks."""
        assert block_id in self.free_block_ids, f"Block {block_id} not in free list"
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count not zero"

        block.reset()
        self.free_block_ids.remove(block_id)
        self.req_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        """Deallocate a block and return it to free list."""
        block = self.blocks[block_id]
        assert (
            block.ref_count == 0
        ), f"Block {block_id} ref_count not zero, cannot deallocate"

        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        block.free()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, num_required_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_required_blocks

    def allocate_blocks(
        self, token_ids: List[int], block_table: List[int] = None
    ) -> tuple[List[int], List[int], int]:
        """Allocate cache blocks for new request with prefix caching support.

        Args:
            token_ids: Input token sequence
            block_table: Existing block_table (for decode phase)

        Returns:
            Tuple of (block_table, slot_mapping, num_cached_tokens)
        """
        if block_table is None:
            block_table = []

        num_tokens = len(token_ids)
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        slot_mapping = []
        num_cached_tokens = 0
        prefix_hash = -1
        cache_miss = False

        for block_idx in range(num_blocks):
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, num_tokens)
            block_tokens = token_ids[start_idx:end_idx]

            # Only full blocks can be hashed for reuse
            if len(block_tokens) == self.block_size:
                prefix_hash = self.compute_hash(block_tokens, prefix_hash)

                # Try to reuse existing block
                if not cache_miss:
                    cached_block_id = self.hash_to_block_id.get(prefix_hash, -1)
                    if (
                        cached_block_id != -1
                        and self.blocks[cached_block_id].token_ids == block_tokens
                    ):
                        # Check if all tokens are cached
                        if num_cached_tokens + self.block_size == len(token_ids):
                            cache_miss = True
                        else:
                            # Reuse successful
                            block = self.blocks[cached_block_id]
                            block.ref_count += 1
                            block_table.append(cached_block_id)
                            num_cached_tokens += self.block_size
                            continue
                    else:
                        cache_miss = True
            else:
                prefix_hash = -1

            # Cannot reuse, allocate new block
            if not self.free_block_ids:
                raise RuntimeError("No available cache blocks")

            new_block_id = self.free_block_ids[0]
            if prefix_hash != -1:
                block = self._allocate_full_block(new_block_id)
                block.update(prefix_hash, block_tokens)
            else:
                block = self._allocate_partial_block(new_block_id)
            block_table.append(new_block_id)

            # Generate slot_mapping
            for i in range(len(block_tokens)):
                slot_mapping.append(new_block_id * self.block_size + i)

        return block_table, slot_mapping, num_cached_tokens

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
            new_block_id = self.free_block_ids[0]
            self._allocate_partial_block(new_block_id)
            block_table.append(new_block_id)

        # Calculate slot
        last_block_id = block_table[-1]
        offset = (num_tokens - 1) % self.block_size
        slot_id = last_block_id * self.block_size + offset

        return block_table, slot_id

    def free_blocks(self, block_table: List[int]):
        """Decrease reference count for all blocks. Blocks with ref_count=0 are not
        immediately freed to allow reuse."""
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
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

    def update_blocks_hash(self, block_table: List[int], num_cached_tokens: int):
        """Register hashes for blocks beyond the locally cached prefix into the lookup table.

        Called on the decode node after receiving KV data from the prefill node,
        so that subsequent requests can hit these blocks via prefix caching.
        Only full blocks (with a valid hash) are registered; partial blocks are skipped.

        Args:
            block_table: Block IDs for the current request.
            num_cached_tokens: Number of locally cached tokens (must be a multiple of
                block_size).
        """
        assert (
            num_cached_tokens % self.block_size == 0
        ), "num_cached_tokens must be multiple of block_size"
        for idx in range(num_cached_tokens // self.block_size, len(block_table)):
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

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def get_total_usable_blocks(self) -> int:
        freeable_used_blocks = sum(
            1 for bid in self.used_block_ids if self.blocks[bid].ref_count == 0
        )
        return len(self.free_block_ids) + freeable_used_blocks

    def __repr__(self):
        return (
            f"BlockManager(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"free={len(self.free_block_ids)}, used={len(self.used_block_ids)})"
        )

    def get_computed_blocks(self, token_ids: List[int]) -> tuple[List[int], int]:
        """Find locally cached prefix blocks for the given token sequence.

        The last token is never matched, as it must be recomputed to obtain logits.

        Returns:
            A tuple of (cached_block_table, num_cached_tokens):
            - cached_block_table: List of matched block IDs (each with ref_count incremented).
            - num_cached_tokens: Number of cached tokens (always a multiple of block_size).
        """
        num_tokens = len(token_ids)
        max_cache_hit_length = num_tokens - 1  # last token must be recomputed

        cached_block_table = []
        num_cached_tokens = 0
        prefix_hash = -1

        num_full_blocks = max_cache_hit_length // self.block_size

        for block_idx in range(num_full_blocks):
            start_idx = block_idx * self.block_size
            end_idx = start_idx + self.block_size
            block_tokens = token_ids[start_idx:end_idx]

            prefix_hash = self.compute_hash(block_tokens, prefix_hash)
            cached_block_id = self.hash_to_block_id.get(prefix_hash, -1)

            if (
                cached_block_id != -1
                and self.blocks[cached_block_id].token_ids == block_tokens
            ):
                self.blocks[cached_block_id].ref_count += 1
                cached_block_table.append(cached_block_id)
                num_cached_tokens += self.block_size
            else:
                break

        return cached_block_table, num_cached_tokens

    def allocate_slots(
        self,
        token_ids: List[int],
        num_new_tokens: int,
        num_local_computed_tokens: int = 0,
        cached_block_table: List[int] = None,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> tuple[List[int], List[int]] | None:
        """Allocate KV cache slots for a request (PD-disaggregation aware).

        Note: Requires that the underlying attention kernel writes KV cache before
        reading it (write-before-read ordering).

        Args:
            token_ids: Complete token sequence for the request.
            num_new_tokens: Number of tokens to compute in this step.
            num_local_computed_tokens: Locally cached prefix token count.
            cached_block_table: Already-matched local block IDs.
            num_external_computed_tokens: Token count matched on the remote prefill node.
                Blocks are still allocated locally to receive the transferred KV data.
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

        total_computed = num_local_computed_tokens + num_external_computed_tokens
        total_tokens = total_computed + num_new_tokens

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

            new_block_id = self.free_block_ids[0]

            if is_full_block:
                prefix_hash = self.compute_hash(block_tokens, prefix_hash)
                block = self._allocate_full_block(new_block_id)
                block.update(prefix_hash, block_tokens)
            else:
                block = self._allocate_partial_block(new_block_id)

            block_table.append(new_block_id)

        for tok_idx in range(total_computed, total_tokens):
            blk_idx = tok_idx // self.block_size
            blk_offset = tok_idx % self.block_size
            slot_mapping.append(block_table[blk_idx] * self.block_size + blk_offset)

        if delay_cache_blocks:
            for block_id in list(self.req_block_ids):
                self.used_block_ids.add(block_id)
            self.req_block_ids.clear()
        else:
            self.reset_req_blocks()

        return block_table, slot_mapping
