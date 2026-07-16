#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Attention {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, size_t);
    static void execute(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor attention(Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos);
void attention_(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos);
} // namespace infinicore::op
