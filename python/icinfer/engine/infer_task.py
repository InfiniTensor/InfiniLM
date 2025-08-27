from typing import List
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
from icinfer.engine.libinfinicore_infer import (
    KVCacheCStruct,
)




class InferTask:
    def __init__(self, id, tokens, max_tokens, temperature, topk, topp, end_tokens):
        self.id = id
        self.finish_reason = None
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.topk = topk
        self.topp = topp
        self.end_tokens = end_tokens
        self._kv_cache = None
        self.pos = 0

    def bind_kvcache(self, kv_cache, pos=0):
        self._kv_cache = kv_cache
        self.pos = pos
        self.tokens = self.tokens[pos:]

    def release_kvcache(self):
        cache = self._kv_cache
        self._kv_cache = None
        return cache

    def kvcache(self):
        return self._kv_cache

    def next(self, out_token):
        if self._kv_cache is not None:
            self._kv_cache.update_tokens(self.tokens, self.pos)

        self.pos += len(self.tokens)
        if out_token == None or out_token in self.end_tokens:
            self.finish_reason = "stop"
        elif self.pos >= self.max_tokens:
            self.finish_reason = "length"
        else:
            self.tokens = [out_token]


class InferBatchedTask:
    def __init__(self, tasks: List[InferTask], is_prefill: int=1):
        self.tasks = tasks
        self.nreq = len(tasks)
        self.is_prefill = is_prefill

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.block_tables = POINTER(c_int)()
        self.slot_mapping = POINTER(c_int)()
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.block_tables,
            self.slot_mapping,
            self.temperaturas,
            self.topks,
            self.topps,
            self.is_prefill,
            )


class InferPagedBatchedTask:
    def __init__(self, tasks: List[InferTask], batch_block_tables: list[int]=[], slot_mapping: list[int]=[], paged_kvcache=None, is_prefill: int=1):
        self.tasks = tasks
        self.nreq = len(tasks)
        self.is_prefill = is_prefill
        self.batch_block_tables = batch_block_tables
        self.slot_mapping = slot_mapping

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [paged_kvcache.data()]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)
        self.n_blocks = len(batch_block_tables) # self.nreq * max_block_table_lens

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * 1)(*self.kv_cache_ptrs)
        self.block_tables = (c_int * self.n_blocks)(*batch_block_tables)
        self.slot_mapping = (c_int * self.ntok)(*slot_mapping)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.block_tables,
            self.slot_mapping,
            self.temperaturas,
            self.topks,
            self.topps,
            self.is_prefill,
            )
    
    def input_args_for_logits(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.block_tables,
            self.slot_mapping,
            self.is_prefill,
            )



class KVCache:
    def __init__(self, model):
        self._kvcache = model.create_kv_cache()
        self.tokens = [0 for _ in range(model.max_context_len())]

    def data(self):
        return self._kvcache

    def drop(self, model):
        model.drop_kv_cache(self._kvcache)

    def update_tokens(self, tokens, pos):
        end = pos + len(tokens)
        max_len = len(self.tokens)

        # If overflow, truncate tokens to fit
        if end > max_len:
            tokens = tokens[: max_len - pos]
            end = max_len

        self.tokens[pos:end] = tokens

class PagedKVCache:
    def __init__(self, paged_kvcache):
        self._kvcache = paged_kvcache
        # self.tokens = [0 for _ in range(model.max_context_len())]

    def data(self):
        return self._kvcache

    def drop(self, model):
        model.drop_kv_cache(self._kvcache)
    
    def update_tokens(self, tokens, pos):
        print("PagedKVCache need not to update tokens.")
        pass
