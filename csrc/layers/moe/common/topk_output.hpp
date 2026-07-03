#pragma once

#include "infinicore/tensor.hpp"

namespace infinilm::layers::moe {

struct TopKOutput {
    infinicore::Tensor topk_weights;
    infinicore::Tensor topk_ids;
    infinicore::Tensor router_logits;
};

} // namespace infinilm::layers::moe
