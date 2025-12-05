#pragma once

#include "infinicore/nn/module.hpp"

#include <any>

namespace infinilm {
class InfinilmModel : public infinicore::nn::Module {
public:
    virtual ~InfinilmModel() = default;
    virtual infinicore::Tensor forward(std::vector<std::any>) const = 0;
};
} // namespace infinilm
