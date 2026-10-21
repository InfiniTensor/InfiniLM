#include "kv_cache.hpp"
#include "../execution/submit.hpp"

#include "../engine/distributed/communication_group.hpp"
#include "../utils.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::cache {
namespace {
size_t rank_kv_heads(infinilm::runtime::Size k_dim, infinilm::runtime::Size v_dim,
                     infinilm::runtime::Size k_heads, infinilm::runtime::Size v_heads,
                     size_t tp_size) {
    if (k_dim == 0 || v_dim == 0 || k_heads == 0 || v_heads == 0
        || k_dim != v_dim || k_heads != v_heads) {
        throw std::invalid_argument("KV cache requires positive equal K/V dimensions and head counts");
    }
    if (k_heads < tp_size) {
        if (tp_size % k_heads != 0) {
            throw std::invalid_argument("KV cache replication requires TP divisible by KV heads");
        }
        return 1;
    }
    if (k_heads % tp_size != 0) {
        throw std::invalid_argument("KV cache head count must be divisible by TP");
    }
    return k_heads / tp_size;
}
} // namespace

void validate_cache_runtime(infinilm::runtime::Runtime &runtime,
                            const engine::distributed::RankInfo &rank) {
    runtime.assertOwnerThread();
    if (runtime.device() != rank.device) {
        throw std::invalid_argument("KV cache Runtime and rank devices must agree");
    }
    if (rank.tp_size <= 0) {
        throw std::invalid_argument("cache TP size must be positive");
    }
    const parallel::Topology topology(static_cast<size_t>(rank.tp_size), rank.pp_size, rank.pp_stage);
    (void)topology.world_rank(rank.tp_rank);
    if (runtime.isGraphRecording()
        || runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("KV cache allocation/reset requires inactive graph state");
    }
}

void validate_backend_cache_config(const CacheConfig *config,
                                   backends::AttentionBackend backend) {
    if (config == nullptr) {
        return;
    }
    switch (backend) {
    case backends::AttentionBackend::STATIC_ATTN:
        if (dynamic_cast<const StaticKVCacheConfig *>(config) == nullptr) {
            throw std::invalid_argument("static attention requires StaticKVCacheConfig");
        }
        return;
    case backends::AttentionBackend::PAGED_ATTN:
    case backends::AttentionBackend::FLASH_ATTN:
        if (dynamic_cast<const PagedKVCacheConfig *>(config) == nullptr) {
            throw std::invalid_argument("paged/flash attention requires PagedKVCacheConfig");
        }
        return;
    default:
        throw std::invalid_argument("unsupported attention backend for KV cache");
    }
}

// ==========================
// StaticKVCacheConfig
// ==========================

StaticKVCacheConfig::StaticKVCacheConfig(
    infinilm::runtime::Size _max_batch_size,
    infinilm::runtime::Size _max_cache_len)
    : max_batch_size_(_max_batch_size),
      max_cache_len_(_max_cache_len) {
}

std::unique_ptr<CacheConfig>
StaticKVCacheConfig::unique_copy() const {
    return std::make_unique<StaticKVCacheConfig>(*this);
}

infinilm::runtime::Size
StaticKVCacheConfig::max_batch_size() const {
    return max_batch_size_;
}

infinilm::runtime::Size
StaticKVCacheConfig::max_cache_len() const {
    return max_cache_len_;
}

namespace StaticKVCache {

// ==========================
// StaticKVCache
// ==========================
infinilm::runtime::Tensor create_layer_kv_cache(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    const infinilm::runtime::Size k_dim,
    const infinilm::runtime::Size v_dim,
    const infinilm::runtime::Size num_k_heads,
    const infinilm::runtime::Size num_v_heads,
    const infinilm::runtime::Size max_positional_embedding,
    const infinilm::runtime::DataType dtype,
    const StaticKVCacheConfig &config) {
    validate_cache_runtime(runtime, rank);
    const size_t num_rank_k_heads = rank_kv_heads(k_dim, v_dim, num_k_heads, num_v_heads, static_cast<size_t>(rank.tp_size));

    size_t rank_batch_size = (config.max_batch_size());
    size_t kv_dim = k_dim;

    size_t cache_len = (config.max_cache_len() == std::numeric_limits<infinilm::runtime::Size>::max() || config.max_cache_len() == 0 ? max_positional_embedding : config.max_cache_len());

    // Allocate KV cache
    infinilm::runtime::Tensor kv_cache = infinilm::runtime::Tensor::empty_exact(
        runtime,
        {2,
         rank_batch_size,
         num_rank_k_heads,
         cache_len,
         kv_dim},
        dtype);
    // Both backing and host source survive any submitted-copy failure.
    std::vector<uint8_t> zeros(kv_cache->nbytes(), 0);
    try {
        runtime.memcpyH2D(kv_cache->data(), zeros.data(), kv_cache->nbytes(), false);
        runtime.syncStream();
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }

    return kv_cache;
}
}; // namespace StaticKVCache

// ==========================
// PagedKVCacheConfig
// ==========================
PagedKVCacheConfig::PagedKVCacheConfig(
    size_t num_blocks,
    size_t block_size,
    size_t max_batch_size)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      max_batch_size_(max_batch_size) {
}

std::unique_ptr<CacheConfig>
PagedKVCacheConfig::unique_copy() const {
    return std::make_unique<PagedKVCacheConfig>(*this);
}

size_t
PagedKVCacheConfig::num_blocks() const {
    return num_blocks_;
}

size_t
PagedKVCacheConfig::block_size() const {
    return block_size_;
}

size_t
PagedKVCacheConfig::max_batch_size() const {
    return max_batch_size_;
}

namespace PagedKVCache {
// ==========================
// PagedKVCache
// ==========================
infinilm::runtime::Tensor create_layer_kv_cache(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    infinilm::runtime::Size k_dim,
    infinilm::runtime::Size v_dim,
    infinilm::runtime::Size num_k_heads,
    infinilm::runtime::Size num_v_heads,
    infinilm::runtime::DataType dtype,
    const PagedKVCacheConfig &config) {
    validate_cache_runtime(runtime, rank);
    const size_t num_rank_k_heads = rank_kv_heads(k_dim, v_dim, num_k_heads, num_v_heads, static_cast<size_t>(rank.tp_size));

    size_t kv_dim = k_dim;

    size_t num_blocks_per_layer = config.num_blocks();
    size_t block_size = config.block_size();

    // Canonical FlashAttention cache layout: [K/V, blocks, block, heads, dim].
    infinilm::runtime::Shape kv_shape = {
        2, num_blocks_per_layer, block_size, num_rank_k_heads, k_dim};

    infinilm::runtime::Tensor kv_cache = infinilm::runtime::Tensor::empty_exact(runtime, kv_shape, dtype);
    try {
        if (kv_cache->nbytes() != 0) {
            if (kv_cache->is_contiguous() && !runtime.isGraphRecording()) {
                if (!infini::ops::Fill::valid_output(infinilm::execution::detail::view(kv_cache))) {
                    throw std::invalid_argument("zero-fill output layout or storage is invalid");
                }
                runtime.setDeviceMemoryAsync(kv_cache->data(), 0, kv_cache->nbytes(), runtime.stream());
            } else {
                infinilm::execution::submit<infini::ops::Fill>(
                    runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
                    kv_cache, 0.0, kv_cache);
            }
            INFINILM_TRACE_OP(Zeros, kv_cache);
        }
    } catch (...) {
        // The factory's destination remains alive until the owner drain proves safety.
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
    // The caller retains its existing completion boundary; no per-layer sync.
    return kv_cache;
}
}; // namespace PagedKVCache

} // namespace infinilm::cache
