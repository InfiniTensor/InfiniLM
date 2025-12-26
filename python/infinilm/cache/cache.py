from infinilm.lib import _infinilm


class CacheConfig(_infinilm.CacheConfig):
    def __init__(self):
        raise NotImplementedError(
            "CacheConfig is an abstract class. Please use a subclass of CacheConfig."
        )


class StaticKVCacheConfig(CacheConfig, _infinilm.StaticKVCacheConfig):
    def __init__(self, max_batch_size: int = 1, max_cache_len: int = 0):
        _infinilm.StaticKVCacheConfig.__init__(self, max_batch_size, max_cache_len)


class PagedKVCacheConfig(CacheConfig, _infinilm.PagedKVCacheConfig):
    def __init__(
        self,
        max_kv_memory_bytes: int,
        block_size: int = 16,
    ):
        _infinilm.PagedKVCacheConfig.__init__(
            self,
            max_kv_memory_bytes,
            block_size,
        )
