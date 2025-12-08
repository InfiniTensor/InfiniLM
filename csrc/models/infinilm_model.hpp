#pragma once

#include "infinicore/nn/module.hpp"

#include <any>

namespace infinilm {
class InfinilmModel : public infinicore::nn::Module {
public:
    virtual ~InfinilmModel() = default;
    virtual infinicore::Tensor forward(std::vector<std::any>) const = 0;
    // Optional: reset cache; default no-op for models without cache
    virtual void reset_cache(size_t /*pos*/ = 0) {}
};
} // namespace infinilm
