// cache_config.hpp (modified)
#pragma once

#include <cstddef>
#include <string>

namespace infinilm::cache {

/**
 * @enum CacheType
 * @brief Enumeration of supported cache types
 */
enum class CacheType {
    DYNAMIC, ///< Dynamic KV cache (grows as needed)
    PAGED,   ///< Paged KV cache (for paged attention)
};

enum class CacheResetMode {
    PRESERVE, // Keep cache memory, only reset positions
    RECREATE  // Recreate cache with new configuration
};

struct CacheConfig {
    CacheType type = CacheType::DYNAMIC;
    size_t num_layers = 0;
    size_t max_kv_cache_length = 0;
    size_t initial_capacity = 1024; // Initial cache capacity in tokens
    size_t initial_batch_size = 1;  // Initial batch size for cache allocation
    float growth_factor = 2.0f;     // Cache growth factor when resizing
    CacheResetMode reset_mode = CacheResetMode::PRESERVE;

    // Constructor
    CacheConfig() = default;
    CacheConfig(CacheType t, size_t nl = 32, size_t mpe = 4096)
        : type(t), num_layers(nl), max_kv_cache_length(mpe) {}

    bool operator==(const CacheConfig &other) const {
        return type == other.type && num_layers == other.num_layers && max_kv_cache_length == other.max_kv_cache_length && initial_capacity == other.initial_capacity && initial_batch_size == other.initial_batch_size && growth_factor == other.growth_factor;
    }

    bool operator!=(const CacheConfig &other) const {
        return !(*this == other);
    }
};

} // namespace infinilm::cache
