#pragma once
#include "../engine/distributed/distributed.hpp"
#include "infinicore/tensor.hpp"

namespace infinilm::cache {
class Cache {
public:
    Cache() = default;
    virtual ~Cache() {}
};

class CacheConfig {
public:
    CacheConfig() = default;
    virtual ~CacheConfig() {}

    virtual std::unique_ptr<CacheConfig> unique_copy() const = 0;
};
} // namespace infinilm::cache
