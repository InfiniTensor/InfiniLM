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
    drop_kv_cache,
    infer_batch,
    forward_batch,
)

from icinfer.layers.sampler import Sampler
from icinfer.utils.context import set_context, get_context, reset_context
# from icinfer.utils.loader import load_model
from icinfer.utils.jiuge_weights_loader import load_model
from icinfer.engine.infer_task import InferTask, InferBatchedTask, InferPagedBatchedTask, PagedKVCache


# infinicore infer
from typing import List, Sequence
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import time
import math
from icinfer.engine.infer_task import InferTask, KVCache


logger = logging.getLogger(__name__)


class ModelRunner:

    def __init__(self, config: Config, device: DeviceType, rank: int, event: Event | list[Event]):
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

        # dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # torch.cuda.set_device(rank)
        # default_dtype = torch.get_default_dtype()
        # torch.set_default_dtype(hf_config.torch_dtype)
        # torch.set_default_device("cuda")

        # model_set = {
        #     "qwen3": Qwen3ForCausalLM,
        #     "fm9g7b": FM9GForCausalLM,
        # }
        # ModelForCausalLm = model_set[hf_config.model_type]
        # self.model = ModelForCausalLm(hf_config)
        # load_model(self.model, config.model)
        
        self.model, self.meta = load_model(self.config, device)
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        #             model_dir_path, trust_remote_code=True
        #         )
        
        eos_token_id = self.hf_config.eos_token_id
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )

        # self.sampler = Sampler()
        # self.warmup_model()
        # TODO 暂时先关掉
        if self.enable_paged_attn:
            self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # torch.set_default_device("cpu")
        # torch.set_default_dtype(default_dtype)

        # if self.world_size > 1:
        #     if rank == 0:
        #         self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        #         dist.barrier()
        #     else:
        #         dist.barrier()
        #         self.shm = SharedMemory(name="nanovllm")
        #         self.loop()

    def exit(self):
        # if self.world_size > 1:
        #     self.shm.close()
        #     dist.barrier()
        #     if self.rank == 0:
        #         self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        self.destroy()
        # dist.destroy_process_group()

    def __del__(self):
        self.destroy()

    def destroy(self):
        """
        在程序退出时，安全地释放 C++ 侧的资源。
        """
        if hasattr(self, 'kv_cache') and self.kv_cache:
            print("drop_kv_cache")
            drop_kv_cache(self.model, self.kv_cache.data())
            self.kv_cache = None
        if hasattr(self, 'model') and self.model:
            destroy_jiuge_model(self.model)
            self.model = None
        
            logger.info("ModelRunner model resources have been released.")

    # def loop(self):
    #     while True:
    #         method_name, args = self.read_shm()
    #         self.call(method_name, *args)
    #         if method_name == "exit":
    #             break

    # def read_shm(self):
    #     assert self.world_size > 1 and self.rank
    #     self.event.wait()
    #     n = int.from_bytes(self.shm.buf[0:4], "little")
    #     method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    #     self.event.clear()
    #     return method_name, args

    # def write_shm(self, method_name, *args):
    #     assert self.world_size > 1 and not self.rank
    #     data = pickle.dumps([method_name, *args])
    #     n = len(data)
    #     self.shm.buf[0:4] = n.to_bytes(4, "little")
    #     self.shm.buf[4:n+4] = data
    #     for event in self.event:
    #         event.set()

    def call(self, method_name, *args):
        # if self.world_size > 1 and self.rank == 0:
        #     self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()
    
    # def _calculate_num_blocks(self, num_kv_heads: int) -> int:
    #     config = self.config
    #     hf_config = config.hf_config
    #     gpu_memory_utilization = config.gpu_memory_utilization
        
    #     free, total = torch.cuda.mem_get_info()
    #     used = total - free
    #     # todo torch.cuda需要用一个什么来替代这部分
    #     peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    #     current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    #     block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
    #     num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
    #     assert num_kvcache_blocks > 0
    #     return num_kvcache_blocks

    def allocate_kv_cache(self):
        kv_cache = self.create_paged_kv_cache(self.config.max_kvcache_tokens)
        self.kv_cache = PagedKVCache(kv_cache)
        print("kvcache allocated ")
        # config = self.config
        # hf_config = config.hf_config
        # num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # config.num_kvcache_blocks = self._calculate_num_blocks(num_kv_heads)
        # self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        # layer_id = 0
        # for module in self.model.modules():
        #     if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        #         module.k_cache = self.kv_cache[0, layer_id]
        #         module.v_cache = self.kv_cache[1, layer_id]
        #         layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        padded_lists_generator = (
            (seq.block_table + [0] * (max_len - len(seq.block_table))) 
            for seq in seqs
        )
        block_tables_flat = list(itertools.chain.from_iterable(padded_lists_generator))
        # block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
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
            input_ids.extend(seq[seq.num_cached_tokens:])
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
        # if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
        block_tables = self.prepare_block_tables(seqs)
        # input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        # return input_ids, positions
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
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        # input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return block_tables, slot_mapping

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
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
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])
    

    # def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    #     input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    #     temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    #     logits = self.run_model(input_ids, positions, is_prefill)
    #     token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    #     reset_context()
    #     return token_ids

    # @torch.inference_mode()
    # def capture_cudagraph(self):
    #     config = self.config
    #     hf_config = config.hf_config
    #     max_bs = min(self.config.max_num_seqs, 512)
    #     max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    #     input_ids = torch.zeros(max_bs, dtype=torch.int64)
    #     positions = torch.zeros(max_bs, dtype=torch.int64)
    #     slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    #     context_lens = torch.zeros(max_bs, dtype=torch.int32)
    #     block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    #     outputs = torch.zeros(max_bs, hf_config.hidden_size)
    #     self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    #     self.graphs = {}
    #     self.graph_pool = None

    #     for bs in reversed(self.graph_bs):
    #         graph = torch.cuda.CUDAGraph()
    #         set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
    #         outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
    #         with torch.cuda.graph(graph, self.graph_pool):
    #             outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
    #         if self.graph_pool is None:
    #             self.graph_pool = graph.pool()
    #         self.graphs[bs] = graph
    #         torch.cuda.synchronize()
    #         reset_context()

    #     self.graph_vars = dict(
    #         input_ids=input_ids,
    #         positions=positions,
    #         slot_mapping=slot_mapping,
    #         context_lens=context_lens,
    #         block_tables=block_tables,
    #         outputs=outputs,
    #     )

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
    

    # infinifore infer
    def max_context_len(self):
        return self.meta.dctx
        # return self.config.max_model_len

    def create_kv_cache(self):
        return create_kv_cache(self.model)

    def drop_kv_cache(self, kv_cache):
        drop_kv_cache(self.model, kv_cache)

    def create_paged_kv_cache(self, max_kvcache_tokens):
        return create_paged_kv_cache(self.model, max_kvcache_tokens)
    
    # @torch.inference_mode()
    # def batch_infer_one_round(self, tasks: List[InferTask]):
    #     output = (c_uint * len(tasks))()
    #     batch_inputs = InferBatchedTask(tasks)
    #     infer_batch(
    #         self.model,
    #         *(batch_inputs.input_args()),
    #         output,
    #     )
    #     return list(output)

    def batch_infer_one_round(self, tasks: List[InferTask], is_prefill: int, batch_block_tables: list[int], slot_mapping: list[int]):
        output = (c_uint * len(tasks))()
        batch_inputs = None
        if self.enable_paged_attn:
            batch_inputs = InferPagedBatchedTask(tasks, batch_block_tables, slot_mapping, self.kv_cache, is_prefill)
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
        # input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # logits = self.run_model(input_ids, positions, is_prefill)
        # token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # reset_context()
        batch_block_tables, slot_mapping = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        tasks = [seq.infer_task for seq in seqs]
        token_ids = self.batch_infer_one_round(tasks, is_prefill, batch_block_tables, slot_mapping)

        return token_ids
