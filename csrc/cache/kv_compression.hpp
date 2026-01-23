#pragma once

#include "kv_cache.hpp"

#include <cstdint>
#include <string>

namespace infinilm::cache {

struct KVCompressionConfig {
    bool enable = false;
    uint32_t compression_factor = 1;
    uint32_t min_seq_len = 0;
    uint32_t image_kv_len = 0;
    std::string weight_path;
};

uint32_t compress_kv_cache_inplace(
    StaticKVCache &cache,
    uint32_t seq_len,
    size_t batch_size,
    const KVCompressionConfig &cfg);

} // namespace infinilm::cache
