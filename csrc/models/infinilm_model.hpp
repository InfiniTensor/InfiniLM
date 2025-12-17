#pragma once

#include "infinicore/nn/module.hpp"

#include "../cache/cache.hpp"

#include <any>

namespace infinilm {
class InfinilmModel : public infinicore::nn::Module {
public:
    struct Config {
        std::string model_type;

        virtual ~Config() = default;
    };

    virtual ~InfinilmModel() = default;
    virtual infinicore::Tensor forward(std::vector<std::any>) const = 0;
    // Optional: reset cache; default no-op for models without cache
    virtual void reset_cache(size_t pos = 0) {}
    virtual void reset_cache(const cache::CacheConfig &new_config, size_t pos = 0) = 0;
};
} // namespace infinilm
