import asyncio
from typing import List


class KVCachePoolItem:
    def __init__(self, model):
        self.kvcache = model.create_kv_cache()
        self.tokens = [0 for _ in range(model.max_context_len())]

    def drop(self, model):
        model.drop_kv_cache(self.kvcache)


class KVCachePool:
    def __init__(self, model, max_caches: int = 32):
        self.max_caches = max_caches
        self.model = model
        self._available: List[KVCachePoolItem] = [KVCachePoolItem(self.model)]
        self.num_caches = 1
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._shutdown = False

    async def acquire(self, infer_task):
        async with self._not_empty:
            while True:
                if self._shutdown:
                    raise RuntimeError("KVCachePool is shutting down; cannot acquire new cache.")
                if len(self._available) == 0:
                    if self.num_caches < self.max_caches:
                        self.num_caches += 1
                        return infer_task.bind_kvcache(KVCachePoolItem(self.model), 0)
                    else:
                        await self._not_empty.wait()
                else:
                    max_match, max_match_index = self.find_most_matching_cache(
                        infer_task.tokens
                    )
                    kvcache = self._available.pop(max_match_index)
                    return infer_task.bind_kvcache(kvcache, max_match)

    async def release(self, infer_task):
        async with self._not_empty:
            self._available.append(infer_task._kv_cache_pool_item)
            self._not_empty.notify()

    def find_most_matching_cache(self, tokens: List[int]):
        max_match = 0
        max_match_index = 0
        
        def first_different_index(a_, b_):
            for i_, (x_, y_) in enumerate(zip(a_, b_)):
                if x_ != y_:
                    return i_
            return min(len(a_), len(b_))
        
        for i, kvcache in enumerate(self._available): 
            common_elements = first_different_index(tokens, kvcache.tokens)
            if common_elements > max_match:
                max_match = common_elements
                max_match_index = i

        # max match should always be less then input tokens length
        return (min(max_match, len(tokens) - 1), max_match_index)

    async def finalize(self):
        async with self._not_empty:
            self._shutdown = True
            while len(self._available) < self.num_caches:
                await self._not_empty.wait()

            # All caches are now available
            for kvcache in self._available:
                if kvcache is not None:
                    kvcache.drop(self.model)

            self._available.clear()
            self.max_caches = 0
            self.num_caches = 0
