from collections import deque

from icinfer.config import Config
from icinfer.engine.sequence import Sequence, SequenceStatus
from icinfer.engine.block_manager import BlockManager
from icinfer.engine.infer_task import KVCache


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()


    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], int]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        is_prefill = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            is_prefill = 1
            return scheduled_seqs, is_prefill

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        # print(f"is_prefill: {is_prefill}, schedule over.\n")
        return scheduled_seqs, is_prefill

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    #     for seq, token_id in zip(seqs, token_ids):
    #         seq.append_token(token_id)
    #         if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
    #             seq.status = SequenceStatus.FINISHED
    #             self.block_manager.deallocate(seq)
    #             self.running.remove(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[KVCache]:
        drop_kvcache_list = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                drop_kvcache_list.append(seq.infer_task.release_kvcache())
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return drop_kvcache_list

    @property
    def block_size(self):
        return self.block_manager.block_size
