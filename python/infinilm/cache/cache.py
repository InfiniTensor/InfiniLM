from infinilm.lib import _infinilm
import infinicore
from ..modeling_utils import parse_dtype


class CacheConfig(_infinilm.CacheConfig):
    def __init__(self):
        raise NotImplementedError(
            "CacheConfig is an abstract class. Please use a subclass of CacheConfig."
        )


class StaticKVCacheConfig(CacheConfig, _infinilm.StaticKVCacheConfig):
    def __init__(
        self,
        max_batch_size: int = 1,
        max_cache_len: int = 0,
        kv_cache_dtype=None,
    ):
        if isinstance(kv_cache_dtype, str):
            _infinilm.StaticKVCacheConfig.__init__(
                self,
                max_batch_size,
                max_cache_len,
                parse_dtype(kv_cache_dtype)._underlying,
            )
        elif isinstance(kv_cache_dtype, infinicore.dtype.dtype):
            _infinilm.StaticKVCacheConfig.__init__(
                self, max_batch_size, max_cache_len, kv_cache_dtype._underlying
            )
        else:
            _infinilm.StaticKVCacheConfig.__init__(
                self, max_batch_size, max_cache_len, kv_cache_dtype
            )


class PagedKVCacheConfig(CacheConfig, _infinilm.PagedKVCacheConfig):
    def __init__(
        self,
        num_blocks: int,
        block_size: int = 256,
        kv_cache_dtype=None,
    ):
        if isinstance(kv_cache_dtype, str):
            _infinilm.PagedKVCacheConfig.__init__(
                self, num_blocks, block_size, parse_dtype(kv_cache_dtype)._underlying
            )
        elif isinstance(kv_cache_dtype, infinicore.dtype.dtype):
            _infinilm.PagedKVCacheConfig.__init__(
                self, num_blocks, block_size, kv_cache_dtype._underlying
            )
        else:
            _infinilm.PagedKVCacheConfig.__init__(
                self, num_blocks, block_size, kv_cache_dtype
            )
