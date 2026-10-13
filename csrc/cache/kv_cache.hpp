#pragma once

#include "../backends/attention_backends.hpp"

#include "base_cache.hpp"
#include <runtime/dtype.hpp>

#include <limits>
#include <memory>

namespace infinilm::runtime { class Runtime; }
namespace infinilm::engine::distributed { struct RankInfo; }

namespace infinilm::cache {
void validate_cache_runtime(infinilm::runtime::Runtime &runtime,
                            const engine::distributed::RankInfo &rank);
void validate_backend_cache_config(const CacheConfig *config,
                                   backends::AttentionBackend backend);
class StaticKVCacheConfig final : public CacheConfig {
public:
    StaticKVCacheConfig(
        infinilm::runtime::Size _max_batch_size = 1,
        infinilm::runtime::Size _max_cache_len = std::numeric_limits<infinilm::runtime::Size>::max());

    std::unique_ptr<CacheConfig> unique_copy() const override;
    infinilm::runtime::Size max_batch_size() const;
    infinilm::runtime::Size max_cache_len() const;

private:
    infinilm::runtime::Size max_batch_size_;
    infinilm::runtime::Size max_cache_len_;
};

namespace StaticKVCache {

infinilm::runtime::Tensor create_layer_kv_cache(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    infinilm::runtime::Size k_dim,
    infinilm::runtime::Size v_dim,
    infinilm::runtime::Size num_k_heads,
    infinilm::runtime::Size num_v_heads,
    infinilm::runtime::Size max_positional_embedding,
    infinilm::runtime::DataType dtype,
    const StaticKVCacheConfig &config);

} // namespace StaticKVCache

class PagedKVCacheConfig final : public CacheConfig {
public:
    PagedKVCacheConfig(
        size_t num_blocks,
        size_t block_size = 256,
        size_t max_batch_size = 1);

    std::unique_ptr<CacheConfig> unique_copy() const override;
    size_t num_blocks() const;
    size_t block_size() const;
    size_t max_batch_size() const;

private:
    size_t num_blocks_;
    size_t block_size_;
    size_t max_batch_size_;
};

namespace PagedKVCache {
infinilm::runtime::Tensor create_layer_kv_cache(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    infinilm::runtime::Size k_dim,
    infinilm::runtime::Size v_dim,
    infinilm::runtime::Size num_k_heads,
    infinilm::runtime::Size num_v_heads,
    infinilm::runtime::DataType dtype,
    const PagedKVCacheConfig &config);

} // namespace PagedKVCache

} // namespace infinilm::cache
