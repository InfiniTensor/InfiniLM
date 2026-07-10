#include "ernie4_5_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace infinilm::models::ernie4_5_vl {

Ernie45Attention::Ernie45Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t layer_idx,
                                   const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get_head_dim();
    rotary_dim_ = model_config->get_rotary_dim();

    const auto &dtype = model_config->get_dtype();
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    const bool use_bias = model_config->get_or<bool>("use_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || (0 != (total_num_kv_heads % static_cast<size_t>(tp_size)))) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    num_key_value_heads_ = total_num_kv_heads / static_cast<size_t>(tp_size);

    auto quantization_method = model_config->get_quantization_method();
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, total_num_heads * head_dim_, quantization_method, use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, total_num_kv_heads * head_dim_, quantization_method, use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, total_num_kv_heads * head_dim_, quantization_method, use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method, use_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                                                          kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Ernie45Attention::forward(const infinicore::Tensor &positions,
                                             const infinicore::Tensor &hidden_states) const {
    if (infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Ernie45Attention::forward_static_(const infinicore::Tensor &position_ids,
                                                     const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    auto q = q_proj_->forward(hidden_states_mutable)->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k = k_proj_->forward(hidden_states_mutable)->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v = v_proj_->forward(hidden_states_mutable)->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    infinicore::Tensor pos_ids_for_rope = position_ids;
    const auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() == 3) {
        // ERNIE VL uses 3D RoPE for temporal/height/width positions. InfiniCore currently
        // exposes 1D RoPE here, so text-only runs use the temporal/text axis as a fallback.
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}, {2, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() != 1) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: unsupported position_ids shape");
    }

    auto q_rotary = q->narrow({{3, 0, rotary_dim_}});
    auto k_rotary = k->narrow({{3, 0, rotary_dim_}});
    rotary_emb_->forward(q_rotary, pos_ids_for_rope, true);
    rotary_emb_->forward(k_rotary, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q, k, v);
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Ernie45Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto q = q_proj_->forward(hidden_states_mutable)->view({seq_len, num_attention_heads_, head_dim_});
    auto k = k_proj_->forward(hidden_states_mutable)->view({seq_len, num_key_value_heads_, head_dim_});
    auto v = v_proj_->forward(hidden_states_mutable)->view({seq_len, num_key_value_heads_, head_dim_});

    infinicore::Tensor pos_ids_for_rope = position_ids;
    const auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() == 3) {
        pos_ids_for_rope = position_ids->narrow({{0, 0, 1}, {2, 0, 1}})->view({pos_shape[1]});
    } else if (pos_shape.size() != 1) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::Ernie45Attention: unsupported position_ids shape");
    }

    auto q_rotary = q->narrow({{2, 0, rotary_dim_}});
    auto k_rotary = k->narrow({{2, 0, rotary_dim_}});
    rotary_emb_->forward(q_rotary, pos_ids_for_rope, true);
    rotary_emb_->forward(k_rotary, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q, k, v);
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::ernie4_5_vl
