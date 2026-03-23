#include "flashinfer_attn.hpp"
#include "../../../utils.hpp"
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

namespace infinilm::layers::attention::backends {

FlashInferAttentionImpl::FlashInferAttentionImpl(size_t num_heads,
                                                 size_t head_size,
                                                 float scale,
                                                 size_t num_kv_heads,
                                                 size_t layer_idx)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size) {}

infinicore::Tensor FlashInferAttentionImpl::forward(void *layer,
                                                    const infinicore::Tensor &query,
                                                    const infinicore::Tensor &key,
                                                    const infinicore::Tensor &value,
                                                    std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                                                    const infinilm::InfinilmModel::Input &attn_metadata) const {
    (void)layer;
    (void)key;
    (void)value;
    (void)kv_cache;
    (void)attn_metadata;
    spdlog::error("FlashInferAttentionImpl is not supported");
    return query;
}

} // namespace infinilm::layers::attention::backends
