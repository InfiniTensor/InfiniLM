from collections.abc import Sequence
from typing import List

from .cache_manager import BlockManage


class SlidingWindowBlockManager:
    """Paged slots for a cache whose old logical blocks become unreachable."""

    def __init__(self, num_blocks: int, block_size: int, sliding_window: int):
        if sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        self.manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.block_size = block_size
        self.sliding_window = sliding_window

    def _allocate_block(self) -> int:
        if not self.manager.can_allocate(1) and not self.manager.try_free_blocks(1):
            raise RuntimeError("No available sliding-window cache blocks")
        return self.manager._allocate_block().block_id

    def _release_block(self, block_id: int) -> None:
        if block_id < 0:
            return
        block = self.manager.blocks[block_id]
        if block.ref_count <= 0:
            return
        block.ref_count -= 1
        if block.ref_count == 0:
            self.manager._deallocate_block(block_id)

    def recycle_before(self, processed_tokens: int, table: List[int]) -> None:
        skipped_tokens = max(0, processed_tokens - self.sliding_window + 1)
        skipped_blocks = skipped_tokens // self.block_size
        if skipped_blocks <= 0:
            return
        first_real_block = 0
        while first_real_block < len(table) and table[first_real_block] < 0:
            first_real_block += 1
        if first_real_block >= skipped_blocks:
            return
        for block_idx in range(first_real_block, skipped_blocks):
            self._release_block(table[block_idx])
            table[block_idx] = -1

    def allocate_slots(
        self,
        num_new_tokens: int,
        num_computed_tokens: int = 0,
        cached_block_table: List[int] | None = None,
    ) -> tuple[List[int], List[int]] | None:
        if num_new_tokens < 0 or num_computed_tokens < 0:
            raise ValueError("token counts must be non-negative")
        table = list(cached_block_table or [])
        total_tokens = num_computed_tokens + num_new_tokens
        end_block = (total_tokens + self.block_size - 1) // self.block_size
        if len(table) > end_block:
            return None
        table.extend([-1] * (end_block - len(table)))
        self.recycle_before(num_computed_tokens, table)

        first_new_block = num_computed_tokens // self.block_size
        try:
            for block_idx in range(first_new_block, end_block):
                if table[block_idx] < 0:
                    table[block_idx] = self._allocate_block()
        except RuntimeError:
            return None

        slots = []
        for token_idx in range(num_computed_tokens, total_tokens):
            block_idx, offset = divmod(token_idx, self.block_size)
            slots.append(table[block_idx] * self.block_size + offset)
        return table, slots

    def append_slot(
        self, block_table: List[int], num_tokens: int
    ) -> tuple[List[int], int]:
        result = self.allocate_slots(1, num_tokens - 1, block_table)
        if result is None:
            raise RuntimeError("No available sliding-window cache blocks")
        table, slots = result
        block_table[:] = table
        return block_table, slots[0]

    def free_table(self, block_table: Sequence[int]) -> None:
        for block_id in block_table:
            self._release_block(block_id)

    def free_blocks(self, block_table: Sequence[int]) -> None:
        self.free_table(block_table)

    def get_num_free_blocks(self) -> int:
        return self.manager.get_num_free_blocks()

    def get_total_usable_blocks(self) -> int:
        return self.manager.get_total_usable_blocks()
