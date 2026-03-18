#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "../../models/infinilm_model.hpp"
#include "../fused_linear.hpp"
#include "attn_base.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace infinilm::models::layers::attention {
class FlashAttention : public Attention<FlashAttention> {
public:
    using Attention<FlashAttention>::Attention;

    infinicore::Tensor attn_calculate(const infinicore::Tensor &q_reshaped,
                                      const infinicore::Tensor &k_reshaped,
                                      const infinicore::Tensor &v_reshaped,
                                      const infinilm::InfinilmModel::Input &input,
                                      std::shared_ptr<infinilm::cache::Cache> kv_cache) const;
};

} // namespace infinilm::models::layers::attention
