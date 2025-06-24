import asyncio
from typing import List


class KVCachePoolItem:
    def __init__(self, model):
        self.kvcache = model.create_kv_cache()
        self.tokens = [0 for _ in range(model.max_context_len())]

    def drop(self, model):
        model.drop_kv_cache(self.kvcache)

    def update_tokens(self, tokens, pos):
        end = pos + len(tokens)
        max_len = len(self.tokens)

        # If overflow, truncate tokens to fit
        if end > max_len:
            tokens = tokens[: max_len - pos]
            end = max_len

        self.tokens[pos:end] = tokens


import threading


class KVCachePool:
    def __init__(self, model, max_caches: int = 32):
        self.max_caches = max_caches
        self.model = model
        self._available: List[KVCachePoolItem] = []
        self.num_caches = len(self._available)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._shutdown = False

    def acquire_sync(self, infer_task):
        with self._not_empty:
            while True:
                if self._shutdown:
                    raise RuntimeError(
                        "KVCachePool is shutting down; cannot acquire new cache."
                    )
                if len(self._available) == 0:
                    if self.num_caches < self.max_caches:
                        self.num_caches += 1
                        print(f"[INFO] Task {infer_task.id} created new KVCachePoolItem")
                        return infer_task.bind_kvcache(KVCachePoolItem(self.model), 0)
                    else:
                        self._not_empty.wait()
                else:
                    max_match, max_match_index = self.find_most_matching_cache(
                        infer_task.tokens
                    )
                    kvcache = self._available.pop(max_match_index)
                    print(
                        f"[INFO] Task {infer_task.id} reused KVCachePoolItem {max_match_index} with {max_match} matches"
                    )
                    return infer_task.bind_kvcache(kvcache, max_match)

    def release_sync(self, infer_task):
        with self._not_empty:
            print(f"[INFO] Task {infer_task.id} returned KVCachePoolItem to pool")
            self._available.append(infer_task._kv_cache_pool_item)
            infer_task._kv_cache_pool_item = None
            self._not_empty.notify()

    async def acquire(self, infer_task):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.acquire_sync, infer_task)

    async def release(self, infer_task):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.release_sync, infer_task)

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
            # print(f"{tokens}")
            # print(f"{kvcache.tokens[:len(tokens)]}")
            if common_elements > max_match:
                max_match = common_elements
                max_match_index = i

        return (min(max_match, len(tokens) - 1), max_match_index)

    def finalize(self):
        with self._not_empty:
            self._shutdown = True
            while len(self._available) < self.num_caches:
                self._not_empty.wait()

            for kvcache in self._available:
                if kvcache is not None:
                    kvcache.drop(self.model)

            self._available.clear()
            self.max_caches = 0
            self.num_caches = 0
