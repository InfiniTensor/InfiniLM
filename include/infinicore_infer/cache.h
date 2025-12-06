#ifndef CACHE_H
#define CACHE_H

#include <infinirt.h>

__C __export struct KVCache *createKVCache(
    size_t nlayers,
    size_t max_len,
    size_t nkvh_,
    size_t dk,
    size_t dv,
    infiniDtype_t dtype,
    infiniDevice_t device,
    int *dev_ids,
    size_t ndev);

__C __export struct KVCache *createPagedKVCache(
    size_t nlayers,
    size_t nkvh_,
    size_t kvcache_block_size,
    size_t max_kvcache_tokens,
    size_t dh,
    infiniDtype_t dtype,
    infiniDevice_t device,
    int *dev_ids,
    size_t ndev);

__C __export struct KVCache *duplicateKVCache(const KVCache *kv_cache, size_t seq_len);

__C __export void dropKVCache(KVCache *kv_cache);

__C __export void dropPagedKVCache(KVCache *kv_cache);

#endif /* CACHE_H */
