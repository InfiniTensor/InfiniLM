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

__C __export struct KVCache *duplicateKVCache(const KVCache *kv_cache, size_t seq_len);

__C __export void dropKVCache(KVCache *kv_cache);

__C __export struct MambaCache *createMambaCache(
    // size_t nlayers,
    size_t batch_size,
    size_t nlinear_attention_layers,
    size_t linear_conv_kernel_dim,
    size_t linear_key_head_dim,
    size_t linear_value_head_dim,
    size_t linear_num_key_heads,
    size_t linear_num_value_heads,
    infiniDtype_t dtype,
    infiniDevice_t device,
    int *dev_ids,
    size_t ndev);

__C __export void dropMambaCache(MambaCache *mamba_cache);

#endif /* CACHE_H */
