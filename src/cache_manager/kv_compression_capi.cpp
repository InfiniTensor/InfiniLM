#include "kv_compression.hpp"

#include "../cache.hpp"
#include "../utils.hpp"

#include "infinicore_infer/kv_compression.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
struct CompressorCacheKey {
    std::string weight_path;
    uint32_t compression_factor = 0;
    uint32_t min_seq_len = 0;
    uint32_t image_kv_len = 0;
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int device_id = 0;

    bool operator==(const CompressorCacheKey &o) const {
        return weight_path == o.weight_path &&
               compression_factor == o.compression_factor &&
               min_seq_len == o.min_seq_len &&
               image_kv_len == o.image_kv_len &&
               device == o.device &&
               device_id == o.device_id;
    }
};

struct CompressorCacheKeyHash {
    size_t operator()(const CompressorCacheKey &k) const noexcept {
        size_t h = std::hash<std::string>{}(k.weight_path);
        auto mix = [&](size_t x) {
            // 64-bit mix
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(std::hash<uint32_t>{}(k.compression_factor));
        mix(std::hash<uint32_t>{}(k.min_seq_len));
        mix(std::hash<uint32_t>{}(k.image_kv_len));
        mix(std::hash<int>{}(static_cast<int>(k.device)));
        mix(std::hash<int>{}(k.device_id));
        return h;
    }
};

// NOTE: Keep the compressor cache alive for the whole process lifetime.
// Python (ctypes) may unload CUDA runtime during interpreter shutdown; running CUDA frees in
// global/static destructors can trigger `cudaErrorCudartUnloading`. Leaking the cache avoids
// freeing CUDA allocations at unload time (OS will reclaim memory on process exit).
std::mutex &cache_mtx() {
    static auto *mtx = new std::mutex();
    return *mtx;
}

std::unordered_map<CompressorCacheKey, std::shared_ptr<Compressor>, CompressorCacheKeyHash> &cache_map() {
    static auto *m = new std::unordered_map<CompressorCacheKey, std::shared_ptr<Compressor>, CompressorCacheKeyHash>();
    return *m;
}
} // namespace

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
    const auto kv_dev = kv_cache->k[0][0]->deviceType();
    const int kv_dev_id = kv_cache->k[0][0]->deviceId();
    RUN_INFINI(infinirtSetDevice(kv_dev, kv_dev_id));

    CompressorCacheKey key;
    key.weight_path = cxx_cfg.weight_path;
    key.compression_factor = cxx_cfg.compression_factor;
    key.min_seq_len = cxx_cfg.min_seq_len;
    key.image_kv_len = cxx_cfg.image_kv_len;
    key.device = kv_dev;
    key.device_id = kv_dev_id;

    std::shared_ptr<Compressor> compressor;
    {
        std::lock_guard<std::mutex> lock(cache_mtx());
        auto &m = cache_map();
        auto it = m.find(key);
        if (it != m.end()) {
            compressor = it->second;
        } else {
            compressor = std::make_shared<Compressor>(cxx_cfg);
            if (!compressor->loadWeights()) {
                return seq_len;
            }
            m.emplace(key, compressor);
        }
    }

    // Compressor is not thread-safe (shared memory pool/context); serialize per cached instance.
    // This keeps behavior deterministic for multi-request Python loops.
    static std::mutex g_run_mtx;
    std::lock_guard<std::mutex> run_lock(g_run_mtx);
    return compressor->compressInplace(*kv_cache, seq_len);
}
