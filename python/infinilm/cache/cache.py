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
        num_blocks: int,
        block_size: int = 16,
    ):
        _infinilm.PagedKVCacheConfig.__init__(
            self,
            num_blocks,
            block_size,
        )


class KVCompressionConfig(_infinilm.KVCompressionConfig):
    def __init__(
        self,
        enable: bool = False,
        compression_factor: int = 1,
        min_seq_len: int = 0,
        image_kv_len: int = 0,
        weight_path: str = "",
    ):
        _infinilm.KVCompressionConfig.__init__(self)
        self.enable = enable
        self.compression_factor = compression_factor
        self.min_seq_len = min_seq_len
        self.image_kv_len = image_kv_len
        self.weight_path = weight_path
