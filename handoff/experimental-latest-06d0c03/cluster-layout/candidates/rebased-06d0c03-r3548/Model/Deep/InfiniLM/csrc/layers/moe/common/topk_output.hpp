#pragma once

#include "runtime/tensor.hpp"

namespace infinilm::layers::moe {

struct TopKOutput {
    infinilm::runtime::Tensor topk_weights;
    infinilm::runtime::Tensor topk_ids;
    infinilm::runtime::Tensor router_logits;
};

} // namespace infinilm::layers::moe
