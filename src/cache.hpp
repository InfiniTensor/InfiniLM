#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>

struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

struct MambaCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> conv_states, ssm_states;
};
