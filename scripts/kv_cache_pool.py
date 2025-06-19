from libinfinicore_infer import KVCache
from ctypes import POINTER
from typing import Dict, List, Tuple, Optional
import threading

class RandSampleArgs:
    def __init__(self, max_tokens, temperature, topk, topp):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.topk = topk
        self.topp = topp

class RequestMeta:
    def __init__(self, id, tokens, args: RandSampleArgs, request, kv_cache: POINTER(KVCache), pos):
        self.id_ = id
        self.tokens_ = tokens
        self.args_ = args
        self.request_ = request
        self.kv_cache_ = kv_cache
        self.pos_ = pos

class KVCachePool:
    def __init__(self, model, max_caches: int = 8):
        self.model = model
        self.max_caches = max_caches
        self.caches: Dict[str, Tuple[int, List[int], POINTER(KVCache)]] = {}
        self.in_use_caches: Dict[str, Tuple[List[int], POINTER(KVCache)]] = {}
        self.lock = threading.Lock()
    
    def find_most_matching_cache(self, tokens: List[int], caches: Dict[str, Tuple[List[int], POINTER(KVCache)]]):
        max_match = 0
        result_pointer = None
        result_tokens = None
    
        set_tokens = set(tokens)
    
        for key, (list_cache_tokens, pointer) in caches.items():
            
            common_elements = len(set_tokens & set(list_cache_tokens))
            if common_elements > max_match:
                max_match = common_elements
                result_tokens = list_cache_tokens
                result_pointer = pointer
    
        return (max_match, result_tokens, result_pointer)
        
    def get_cache(self, tokens:List[int], id: str) -> Optional[Tuple[int, POINTER(KVCache)]]:
        with self.lock: 
            (pos, pre_tokens, kv_cache) = self.find_most_matching_cache(tokens[:-1], self.caches)
            if kv_cache:
                self.in_use_caches[id] = (pos, pre_tokens, kv_cache)
                self.caches.pop(id)
                return (pos, kv_cache)
            elif (len(self.in_use_caches) + len(self.caches)) < self.max_caches:
                new_kv_cache = self.model.create_kv_cache()
                self.in_use_caches[id] = (0, tokens, new_kv_cache)
                return (0, new_kv_cache)
            else:
                return None
            
    def release_cache(self, request: RequestMeta):
        with self.lock:
            self.in_use_caches.pop(request.id_)
            self.caches[request.id_] = (request.tokens_, request.kv_cache_)
