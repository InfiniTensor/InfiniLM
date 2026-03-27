#pragma once

#include "../models/infinilm_model.hpp"

namespace infinilm::engine {

using AttentionMetadata = infinilm::InfinilmModel::Input;

struct ForwardContext {
    AttentionMetadata attn_metadata;
    std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> kv_cache_vec;
};

void set_forward_context(const infinilm::InfinilmModel::Input &input);

ForwardContext &get_forward_context();

} // namespace infinilm::engine
