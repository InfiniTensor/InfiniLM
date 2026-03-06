#ifndef CACHE_H
#define CACHE_H

#include <infinirt.h>

__INFINI_C __export struct KVCache *createKVCache(
    size_t nlayers,
    size_t max_len,
    size_t nkvh_,
    size_t dk,
    size_t dv,
    infiniDtype_t dtype,
    infiniDevice_t device,
    int *dev_ids,
    size_t ndev);

__INFINI_C __export struct KVCache *duplicateKVCache(const KVCache *kv_cache, size_t seq_len);

__INFINI_C __export void dropKVCache(KVCache *kv_cache);

#endif /* CACHE_H */
