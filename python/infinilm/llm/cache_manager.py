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

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def __repr__(self):
        return (
            f"BlockManager(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"free={len(self.free_block_ids)}, used={len(self.used_block_ids)})"
        )
