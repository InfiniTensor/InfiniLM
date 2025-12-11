#pragma once

#include "infinicore/tensor.hpp"

namespace infinilm::cache {

/**
 * @class CacheInterface
 * @brief Abstract interface for all cache types
 */
class CacheInterface {
public:
    virtual ~CacheInterface() = default;

    // Common cache operations
    virtual void update(size_t layer_idx,
                        const infinicore::Tensor &k_new,
                        const infinicore::Tensor &v_new)
        = 0;

    virtual void reset(size_t pos = 0) = 0;
    virtual void reset(CacheConfig &new_config, size_t pos = 0) = 0;
    virtual size_t cache_position(size_t layer_idx) const = 0;
    virtual size_t num_layers() const = 0;
    virtual void *raw_ptr() = 0; // Returns raw pointer for compatibility

    // Type information
    virtual CacheType type() const = 0;
    virtual std::string name() const = 0;

    // Optional advanced features
    virtual void clear() { reset(0); }
    virtual size_t memory_usage() const = 0;
    // virtual bool supports_prefill() = 0;
    // virtual bool supports_incremental() = 0;
};
} // namespace infinilm::cache
