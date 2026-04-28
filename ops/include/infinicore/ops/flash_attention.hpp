#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(FlashAttention, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float, bool);

Tensor flash_attention(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &total_kv_len, float scale, bool is_causal);
void flash_attention_(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &total_kv_len, float scale, bool is_causal);
} // namespace infinicore::op
