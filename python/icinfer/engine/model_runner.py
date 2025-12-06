import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from ctypes import c_uint
from typing import List
import logging
import itertools


from icinfer.config import Config
from icinfer.engine.sequence import Sequence
from icinfer.engine.libinfinicore_infer import (
    JiugeMetaCStruct,
    JiugeWeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType,
    create_jiuge_model,
    destroy_jiuge_model,
    create_kv_cache,
    create_paged_kv_cache,
    drop_paged_kv_cache,
    drop_kv_cache,
    infer_batch,
    forward_batch,
)

from icinfer.layers.sampler import Sampler
from icinfer.utils.context import set_context, get_context, reset_context

# from icinfer.utils.loader import load_model
from icinfer.utils.jiuge_weights_loader import load_model
from icinfer.engine.infer_task import (
    InferTask,
    InferBatchedTask,
    InferPagedBatchedTask,
    PagedKVCache,
)


# infinicore infer
from typing import List, Sequence
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import time
import math
from icinfer.engine.infer_task import InferTask, KVCache


logger = logging.getLogger(__name__)


class ModelRunner:

    def __init__(
        self,
        config: Config,
        device: DeviceType,
        ndev: int,
        rank: int,
        event: Event | list[Event],
    ):
        self.config = config
        self.hf_config = config.hf_config
        self.device = device
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.enable_paged_attn = config.enable_paged_attn
        self.world_size = config.tensor_parallel_size
        self.meta = None
        self.kv_cache = None
        self.rank = rank
        self.event = event
        self.ndev = ndev
        self.dev_ids = (c_int * ndev)(*[i for i in range(ndev)])

        self.model, self.meta = load_model(self.config, device)

        eos_token_id = self.hf_config.eos_token_id
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )

        # self.sampler = Sampler()
        # self.warmup_model()
        # TODO 暂时先关掉
        if self.enable_paged_attn:
            self.allocate_paged_kv_cache()
        else:
            self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

    def exit(self):
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        self.destroy()

    def __del__(self):
        self.destroy()

    def destroy(self):
        """
        在程序退出时，安全地释放 C++ 侧的资源。
        """
        if hasattr(self, "kv_cache") and self.kv_cache:
            print("drop_paged_kv_cache")
            # ！！！待完善的部分，需要后续在处理！！！drop_paged_kv_cache 和 drop_kv_cache
            drop_paged_kv_cache(self.kv_cache.data())
            self.kv_cache = None
        if hasattr(self, "model") and self.model:
            destroy_jiuge_model(self.model)
            self.model = None

            logger.info("ModelRunner model resources have been released.")

    def call(self, method_name, *args):
        # if self.world_size > 1 and self.rank == 0:
        #     self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def allocate_paged_kv_cache(self):
        kv_cache = self.create_paged_kv_cache(
            self.meta.nlayer,
            self.meta.nkvh,
            self.config.kvcache_block_size,
            self.config.max_kvcache_tokens,
            self.meta.dh,
            self.meta.dt_logits,
            self.device,
            self.dev_ids,
            self.ndev,
        )
        self.kv_cache = PagedKVCache(kv_cache)
        print("kvcache allocated ")

    def allocate_kv_cache(self):
        kv_cache = self.create_kv_cache(
            self.meta.nlayer,
            self.config.max_model_len,
            self.meta.nkvh,
            self.meta.dh,
            self.meta.dh,
            self.meta.dt_logits,
            self.device,
            self.dev_ids,
            self.ndev,
        )
        self.kv_cache = KVCache(kv_cache)
        print("kvcache allocated ")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        padded_lists_generator = (
            (seq.block_table + [0] * (max_len - len(seq.block_table))) for seq in seqs
        )
        block_tables_flat = list(itertools.chain.from_iterable(padded_lists_generator))
        return block_tables_flat

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = []
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        block_tables = self.prepare_block_tables(seqs)
        return block_tables, slot_mapping

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        block_tables = self.prepare_block_tables(seqs)
        return block_tables, slot_mapping

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # infinifore infer
    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(
        self, nlayers, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return create_kv_cache(
            nlayers, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_kv_cache(self, kv_cache):
        drop_kv_cache(self.model, kv_cache)

    def create_paged_kv_cache(
        self,
        nlayer,
        nkvh,
        kvcache_block_size,
        max_kvcache_tokens,
        dh,
        dtype,
        device,
        dev_ids,
        ndev,
    ):
        return create_paged_kv_cache(
            nlayer,
            nkvh,
            kvcache_block_size,
            max_kvcache_tokens,
            dh,
            dtype,
            device,
            dev_ids,
            ndev,
        )

    def drop_paged_kv_cache(self, max_kvcache_tokens):
        return drop_paged_kv_cache(self.model, max_kvcache_tokens)

    def batch_infer_one_round(
        self,
        tasks: List[InferTask],
        is_prefill: int,
        batch_block_tables: list[int],
        slot_mapping: list[int],
    ):
        output = (c_uint * len(tasks))()
        batch_inputs = None
        if self.enable_paged_attn:
            batch_inputs = InferPagedBatchedTask(
                tasks, batch_block_tables, slot_mapping, self.kv_cache, is_prefill
            )
        else:
            batch_inputs = InferBatchedTask(tasks, is_prefill)
        infer_batch(
            self.model,
            *(batch_inputs.input_args()),
            self.enable_paged_attn,
            output,
        )
        return list(output)

    def run(self, seqs: list[Sequence], is_prefill: int) -> list[int]:
        batch_block_tables, slot_mapping = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        tasks = [seq.infer_task for seq in seqs]
        token_ids = self.batch_infer_one_round(
            tasks, is_prefill, batch_block_tables, slot_mapping
        )

        return token_ids

    def batch_infer_one_round_for_logits(
        self,
        tasks: List[InferTask],
        is_prefill: int,
        batch_block_tables: list[int],
        slot_mapping: list[int],
    ):
        batch_inputs = None
        if self.enable_paged_attn:
            batch_inputs = InferPagedBatchedTask(
                tasks, batch_block_tables, slot_mapping, self.kv_cache, is_prefill
            )
        else:
            batch_inputs = InferBatchedTask(tasks, is_prefill)
        logits = torch.zeros(
            (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
        )
        forward_batch(
            self.model,
            *(batch_inputs.input_args_for_logits()),
            self.enable_paged_attn,
            logits.data_ptr(),
        )
        return logits, batch_inputs.req_lens_list, batch_inputs.ntok

    def run_for_logits(self, seqs: list[Sequence], is_prefill: int) -> torch.Tensor:
        nll = 0.0
        total_len = 0
        batch_block_tables, slot_mapping = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        tasks = [seq.infer_task for seq in seqs]
        true_tokens = [seq.true_tokens for seq in seqs]
        logits, req_lens_list, ntok = self.batch_infer_one_round_for_logits(
            tasks, is_prefill, batch_block_tables, slot_mapping
        )
        token_ids_none = [None] * len(seqs)

        logits = logits.float()
        token_ids = torch.tensor(true_tokens, dtype=torch.int64).reshape(-1)  # [ntok,]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (ntok, vocab)
        token_logprobs = log_probs[torch.arange(ntok), token_ids]  # (ntok,)

        start = 0
        for l in req_lens_list:
            nll += -token_logprobs[start : start + l].sum().item()
            start += l
        total_len += token_logprobs.numel()

        return nll, total_len, token_ids_none
