"""
KV Cache Manager - Paged Attention block-based cache allocation and management.
"""

from collections import deque
import queue
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
        assert num_blocks > 0 and block_size > 0, (
            "num_blocks and block_size must be positive"
        )
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
        assert block.ref_count == 0, (
            f"Block {block_id} ref_count not zero, cannot deallocate"
        )

        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        block.free()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, num_required_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_required_blocks

    def allocate_blocks(
        self,
        token_ids: List[int],
        block_table: List[int] = None,
        mm_token_index_mappings: List[dict] = None,
    ) -> 'tuple[List[int], List[int], int]':
        """Allocate cache blocks for new request with prefix caching support.

        Args:
            token_ids: Input token sequence
            block_table: Existing block_table (for decode phase)
            mm_token_index_mappings: List of multimodal token index mappings
        Returns:
            Tuple of (block_table, slot_mapping, num_cached_tokens)
        """
        if block_table is None:
            block_table = []

        # Static args
        num_tokens = len(token_ids)
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        num_full_blocks = num_tokens // self.block_size
        remain_tokens = num_tokens % self.block_size
        num_mm_inputs = (
            0 if not mm_token_index_mappings else len(mm_token_index_mappings)
        )

        # Variables
        slot_mapping = []
        num_cached_tokens = 0
        prefix_hash = -1
        cache_miss = False
        mm_start_counter = 0
        mm_caching_queue = queue.Queue(maxsize=len(mm_token_index_mappings))
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
                mm_caching_queue.put((mm_start_counter))
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
                    not mm_caching_queue.empty()
                    and mm_token_index_mappings[mm_caching_queue.queue[0]]["end_index"]
                    < end_idx
                ):
                    mm_caching_queue.get()

            blocks_blueprint.append(
                {"prefix_hash": prefix_hash, "block_id": cached_block_id}
            )

        # If there is one incomplete mm_data, tailing blocks need to fall back until all included mm_data are complete
        if not mm_caching_queue.empty():
            incomplete_mm = mm_token_index_mappings[mm_caching_queue.get()]
            incomplete_mm_start = incomplete_mm[
                "start_index"
            ]  # Fall back until this index is no longer included in the block
            max_blocks_to_reuse = min(
                max_blocks_to_reuse, incomplete_mm_start // self.block_size
            )

        num_cached_tokens = max_blocks_to_reuse * self.block_size

        for block_id in range(num_blocks):
            n_block_tokens = self.block_size

            if block_id < max_blocks_to_reuse:
                # Reuse block
                block = self.blocks[blocks_blueprint[block_id]["block_id"]]
                block.ref_count += 1

            else:
                new_block_id = self.free_block_ids[0]
                if blocks_blueprint[block_id]["prefix_hash"] != -1:
                    start_idx = block_id * self.block_size
                    end_idx = start_idx + self.block_size
                    block_tokens = token_ids[start_idx:end_idx]
                    block = self._allocate_full_block(new_block_id)
                    block.update(
                        blocks_blueprint[block_id]["prefix_hash"], block_tokens
                    )
                else:
                    block = self._allocate_partial_block(new_block_id)
                    n_block_tokens = remain_tokens
                slot_mapping.extend(
                    list(
                        range(
                            block.block_id * self.block_size,
                            block.block_id * self.block_size + n_block_tokens,
                        )
                    )
                )

            block_table.append(block.block_id)

        return block_table, slot_mapping, num_cached_tokens

    def append_slot(
        self, block_table: List[int], num_tokens: int, total_token_ids: List[int] = None
    ) -> 'tuple[List[int], int]':
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
