#include "kv_compression.hpp"

#include "../cache.hpp"
#include "../utils.hpp"

#include "infinicore_infer/kv_compression.h"

#include <algorithm>

__C __export uint32_t
compressKVCacheInplace(struct KVCache *kv_cache, uint32_t seq_len, const KVCompressionConfig *cfg) {
    if (kv_cache == nullptr || cfg == nullptr || cfg->enable == 0) {
        return seq_len;
    }
    if (seq_len == 0) {
        return 0;
    }
    if (kv_cache->k.empty() || kv_cache->k[0].empty() || kv_cache->v.empty() || kv_cache->v[0].empty()) {
        return seq_len;
    }
    if (!kv_cache->k[0][0]) {
        return seq_len;
    }

    CompressionConfig cxx_cfg;
    cxx_cfg.enable = true;
    cxx_cfg.compression_factor = cfg->compression_factor;
    cxx_cfg.min_seq_len = cfg->min_seq_len;
    cxx_cfg.image_kv_len = cfg->image_kv_len;
    if (cfg->weight_path != nullptr) {
        cxx_cfg.weight_path = cfg->weight_path;
    }

    const uint32_t max_seq = static_cast<uint32_t>(kv_cache->k[0][0]->shape()[0]);
    seq_len = std::min<uint32_t>(seq_len, max_seq);

    // Ensure weights are created on the same device as KV.
    RUN_INFINI(infinirtSetDevice(kv_cache->k[0][0]->deviceType(), kv_cache->k[0][0]->deviceId()));

    Compressor compressor(cxx_cfg);
    if (!compressor.loadWeights()) {
        return seq_len;
    }
    return compressor.compressInplace(*kv_cache, seq_len);
}

