"""Paged KV cache allocation and source-agnostic prefix lookup."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Set

from infinilm.llm.prefix_cache import (
    EMPTY_BLOCK_HASH,
    BlockHash,
)


class Block:
    """Control-plane metadata for one physical KV cache page."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash: BlockHash = EMPTY_BLOCK_HASH

    def __repr__(self) -> str:
        return f"Block(id={self.block_id}, ref={self.ref_count}, hash={self.hash})"

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = EMPTY_BLOCK_HASH

    def free(self) -> None:
        self.ref_count = 0
        self.hash = EMPTY_BLOCK_HASH


@dataclass(slots=True)
class SlotAllocation:
    """Private pages and slots allocated for one logical token range."""

    new_blocks: List[int]
    slot_mapping: List[int]


class MambaCacheManager:
    """Manage request ownership of Mamba state cache rows.

    Row 0 is reserved as the permanent zero state. Request-owned rows are
    allocated from [1, num_blocks).
    """

    ZERO_STATE_INDEX = 0

    def __init__(self, num_blocks: int):
        if num_blocks < 2:
            raise ValueError("mamba cache pool size must be at least 2")
        self.num_blocks = num_blocks
        self.free_block_ids: deque[int] = deque(range(1, num_blocks))
        self.used_block_ids: Set[int] = set()

    def can_allocate(self) -> bool:
        return bool(self.free_block_ids)

    def allocate(self) -> int | None:
        if not self.free_block_ids:
            return None
        block_id = self.free_block_ids.popleft()
        self.used_block_ids.add(block_id)
        return block_id

    def free(self, block_id: int | None) -> None:
        if block_id is None or block_id == self.ZERO_STATE_INDEX:
            return
        if block_id not in self.used_block_ids:
            return
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)


class BlockManager:
    """Manage physical paged-cache blocks and published prefix hashes."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_ids: Dict[BlockHash, Set[int]] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()
        self.evictable_block_ids: Set[int] = set()

    def __repr__(self) -> str:
        return (
            f"BlockManager(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"free={len(self.free_block_ids)}, used={len(self.used_block_ids)})"
        )

    def _allocate_block(self) -> Block:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} ref_count not zero"
        block.reset()
        self.used_block_ids.add(block_id)
        return block

    def _remove_block_hash(self, block: Block) -> None:
        if block.hash == EMPTY_BLOCK_HASH:
            return
        block_ids = self.hash_to_block_ids.get(block.hash)
        if block_ids is None or block.block_id not in block_ids:
            raise RuntimeError(
                f"block {block.block_id} hash metadata is missing from the prefix index"
            )
        block_ids.remove(block.block_id)
        if not block_ids:
            del self.hash_to_block_ids[block.hash]
        block.hash = EMPTY_BLOCK_HASH

    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count == 0, (
            f"Block {block_id} ref_count not zero, cannot deallocate"
        )
        self._remove_block_hash(block)
        self.evictable_block_ids.discard(block_id)
        block.free()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, num_required_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_required_blocks

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def get_total_usable_blocks(self) -> int:
        return len(self.free_block_ids) + len(self.evictable_block_ids)

    def get_computed_blocks(
        self,
        block_hashes: Sequence[BlockHash],
        max_cache_hit_tokens: int,
    ) -> tuple[List[int], int]:
        """Pin the longest consecutive cached prefix identified by hashes."""
        max_hit_blocks = min(
            len(block_hashes), max(0, max_cache_hit_tokens) // self.block_size
        )
        cached_block_table: List[int] = []
        for block_idx in range(max_hit_blocks):
            block_hash = block_hashes[block_idx]
            block_ids = self.hash_to_block_ids.get(block_hash)
            if not block_ids:
                break
            block_id = next(iter(block_ids))
            block = self.blocks[block_id]
            assert block.hash == block_hash and block_id in self.used_block_ids
            if block.ref_count == 0:
                self.evictable_block_ids.remove(block_id)
            block.ref_count += 1
            cached_block_table.append(block_id)
        return cached_block_table, len(cached_block_table) * self.block_size

    def allocate_slots(
        self,
        num_new_tokens: int,
        num_computed_tokens: int = 0,
        cached_block_table: List[int] | None = None,
    ) -> tuple[List[int], List[int]] | None:
        """Allocate physical blocks without publishing prefix hashes."""
        if num_new_tokens < 0 or num_computed_tokens < 0:
            raise ValueError("token counts must be non-negative")
        cached_block_table = cached_block_table or []
        block_table = list(cached_block_table)
        cached_tokens = len(block_table) * self.block_size
        if num_computed_tokens < cached_tokens:
            raise ValueError(
                "num_computed_tokens cannot precede the cached block boundary"
            )

        total_tokens = num_computed_tokens + num_new_tokens
        total_blocks = (total_tokens + self.block_size - 1) // self.block_size
        num_blocks_needed = total_blocks - len(block_table)

        if not self.can_allocate(num_blocks_needed):
            if not self.try_free_blocks(num_blocks_needed):
                return None

        for _ in range(num_blocks_needed):
            block_table.append(self._allocate_block().block_id)

        slot_mapping = []
        for token_idx in range(num_computed_tokens, total_tokens):
            block_idx = token_idx // self.block_size
            block_offset = token_idx % self.block_size
            slot_mapping.append(block_table[block_idx] * self.block_size + block_offset)
        return block_table, slot_mapping

    def allocate_slot_range(
        self,
        block_table: Sequence[int],
        start_token: int,
        end_token: int,
    ) -> SlotAllocation:
        """Allocate private pages and slots for ``[start_token, end_token)``.

        The caller owns ``block_table``. This method neither copies nor mutates it;
        only ``new_blocks`` from the returned handle may be appended after the
        scheduler finishes constructing the complete batch.
        """
        required_start_blocks = (start_token + self.block_size - 1) // self.block_size
        if len(block_table) < required_start_blocks:
            raise RuntimeError("block table does not cover the computed token boundary")

        required_blocks = (end_token + self.block_size - 1) // self.block_size
        num_new_blocks = max(required_blocks - len(block_table), 0)
        new_blocks: List[int] = []

        if len(self.free_block_ids) < num_new_blocks:
            if not self.try_free_blocks(num_new_blocks):
                raise RuntimeError(
                    "KV cache capacity invariant violated after admission preflight"
                )

        for _ in range(num_new_blocks):
            new_blocks.append(self._allocate_block().block_id)

        slot_mapping = []
        existing_blocks = len(block_table)
        for token_idx in range(start_token, end_token):
            block_idx, block_offset = divmod(token_idx, self.block_size)
            block_id = (
                block_table[block_idx]
                if block_idx < existing_blocks
                else new_blocks[block_idx - existing_blocks]
            )
            slot_mapping.append(block_id * self.block_size + block_offset)

        return SlotAllocation(new_blocks=new_blocks, slot_mapping=slot_mapping)

    def append_slots(
        self, block_table: List[int], start_num_tokens: int, num_slots: int
    ) -> tuple[List[int], List[int]]:
        """Append contiguous provisional slots for speculative verification."""
        if num_slots < 0:
            raise ValueError("num_slots must be non-negative")
        if num_slots == 0:
            return block_table, []
        if start_num_tokens <= 0:
            raise ValueError("start_num_tokens must be greater than 0")
        expected_blocks = (start_num_tokens + self.block_size - 2) // self.block_size
        if len(block_table) != expected_blocks:
            raise ValueError(
                "start_num_tokens must immediately follow the allocated logical length"
            )

        max_num_tokens = start_num_tokens + num_slots - 1
        required_blocks = (max_num_tokens + self.block_size - 1) // self.block_size
        additional_blocks = max(required_blocks - len(block_table), 0)
        if not self.can_allocate(additional_blocks) and not self.try_free_blocks(
            additional_blocks
        ):
            raise RuntimeError("No available cache blocks")

        for _ in range(additional_blocks):
            block_table.append(self._allocate_block().block_id)

        slots = []
        for num_tokens in range(start_num_tokens, start_num_tokens + num_slots):
            token_idx = num_tokens - 1
            block_idx, block_offset = divmod(token_idx, self.block_size)
            slots.append(block_table[block_idx] * self.block_size + block_offset)
        return block_table, slots

    def truncate_blocks(
        self, block_table: List[int], keep_num_tokens: int
    ) -> List[int]:
        """Release private, unpublished speculative pages past an accepted length.

        The caller must pass a manager-produced block table with unique page IDs.
        """
        if keep_num_tokens <= 0:
            raise ValueError("keep_num_tokens must be greater than 0")
        capacity = len(block_table) * self.block_size
        if keep_num_tokens > capacity:
            raise ValueError(
                f"keep_num_tokens={keep_num_tokens} exceeds block table capacity={capacity}"
            )

        keep_blocks = (keep_num_tokens + self.block_size - 1) // self.block_size
        discarded_block_ids = block_table[keep_blocks:]

        # Validate the complete mutation set first so a malformed speculative
        # table cannot be only partially released.
        for block_id in discarded_block_ids:
            if not 0 <= block_id < self.num_blocks:
                raise RuntimeError(f"invalid provisional block id {block_id}")
            block = self.blocks[block_id]
            if block_id not in self.used_block_ids or block.ref_count != 1:
                raise RuntimeError(
                    f"provisional block {block_id} must be privately owned"
                )
            if block.hash != EMPTY_BLOCK_HASH:
                raise RuntimeError(
                    f"provisional block {block_id} must not be published"
                )

        if keep_num_tokens % self.block_size != 0:
            retained_block_id = block_table[keep_blocks - 1]
            if not 0 <= retained_block_id < self.num_blocks:
                raise RuntimeError(f"invalid retained block id {retained_block_id}")
            retained_block = self.blocks[retained_block_id]
            if (
                retained_block_id not in self.used_block_ids
                or retained_block.ref_count != 1
            ):
                raise RuntimeError(
                    f"retained partial block {retained_block_id} must be privately owned"
                )
            if retained_block.hash != EMPTY_BLOCK_HASH:
                raise RuntimeError(
                    f"retained partial block {retained_block_id} must not be published"
                )

        for block_id in discarded_block_ids:
            block = self.blocks[block_id]
            block.ref_count = 0
            self._deallocate_block(block_id)

        return block_table[:keep_blocks]

    def publish_computed_blocks(
        self,
        block_table: Sequence[int],
        block_hashes: Sequence[BlockHash],
        start_block: int,
        num_computed_tokens: int,
    ) -> int:
        """Publish newly computed full blocks and return the indexed boundary."""
        end_block = min(
            num_computed_tokens // self.block_size,
            len(block_table),
            len(block_hashes),
        )
        if not 0 <= start_block <= end_block:
            raise ValueError(
                f"invalid publish range: start_block={start_block}, end_block={end_block}"
            )
        for block_idx in range(start_block, end_block):
            block_id = block_table[block_idx]
            block = self.blocks[block_id]
            if block.hash != EMPTY_BLOCK_HASH:
                raise RuntimeError(
                    f"published block {block_id} cannot change its prefix hash"
                )

        for block_idx in range(start_block, end_block):
            block_id = block_table[block_idx]
            block = self.blocks[block_id]
            block_hash = block_hashes[block_idx]
            block.hash = block_hash
            self.hash_to_block_ids.setdefault(block_hash, set()).add(block_id)
        return end_block

    def append_slot(
        self, block_table: List[int], num_tokens: int
    ) -> tuple[List[int], int]:
        """Allocate the slot used to compute the latest logical token."""
        if (num_tokens - 1) % self.block_size == 0:
            if not self.free_block_ids and not self.try_free_blocks(1):
                raise RuntimeError("No available cache blocks")
            block_table.append(self._allocate_block().block_id)

        last_block_id = block_table[-1]
        offset = (num_tokens - 1) % self.block_size
        return block_table, last_block_id * self.block_size + offset

    def free_blocks(self, block_table: Sequence[int]) -> None:
        """Release request references while retaining computed blocks for reuse."""
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            assert block.ref_count > 0, "block ref_count must be greater than 0"
            block.ref_count -= 1
            if block.ref_count == 0:
                self.evictable_block_ids.add(block_id)

    def try_free_blocks(self, num_required: int) -> bool:
        """Evict cached pages until the requested free capacity is available."""
        while not self.can_allocate(num_required):
            if not self.evictable_block_ids:
                return False
            self._deallocate_block(next(iter(self.evictable_block_ids)))
        return True

    def update_blocks_slot(
        self, block_table: List[int], num_computed_tokens: int, total_tokens: int
    ) -> List[int]:
        """Build slots for the recomputed suffix after a partial remote load."""
        if num_computed_tokens >= total_tokens:
            return []

        new_slot_mapping = []
        for token_idx in range(num_computed_tokens, total_tokens):
            block_idx = token_idx // self.block_size
            block_offset = token_idx % self.block_size
            new_slot_mapping.append(
                block_table[block_idx] * self.block_size + block_offset
            )
        return new_slot_mapping
