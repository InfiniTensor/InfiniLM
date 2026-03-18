#include "flashinfer_attn.hpp"
#include "../../utils.hpp"
#include "infinicore/io.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/mul.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace infinilm::models::layers::attention {

infinicore::Tensor FlashInferAttention::attn_calculate(const infinicore::Tensor &q_reshaped,
                                                       const infinicore::Tensor &k_reshaped,
                                                       const infinicore::Tensor &v_reshaped,
                                                       const infinilm::InfinilmModel::Input &input,
                                                       std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
    throw std::runtime_error("FlashInferAttention is not supported");
    return q_reshaped;
}

} // namespace infinilm::models::layers::attention
