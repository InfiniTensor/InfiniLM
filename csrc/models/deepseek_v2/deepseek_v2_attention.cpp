#include "deepseek_v2_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"
#include "deepseek_v2_utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/pad.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v2 {

DeepseekV2Attention::DeepseekV2Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    v_head_dim_ = model_config->get<size_t>("v_head_dim");

    const auto &dtype{model_config->get_dtype()};
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const size_t kv_lora_rank = model_config->get<size_t>("kv_lora_rank");
    const bool attention_bias = model_config->get_or<bool>("attention_bias", false);
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    if ((total_num_heads < static_cast<size_t>(tp_size)) || (total_num_heads % static_cast<size_t>(tp_size) != 0)) {
        throw std::runtime_error("DeepseekV2Attention: num_attention_heads must be divisible by tp_size");
    }
    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;

    auto quantization_method = model_config->get_quantization_method();
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, total_num_heads * q_head_dim_, quantization_method, false, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(kv_a_proj_with_mqa, hidden_size_, kv_lora_rank + qk_rope_head_dim_, attention_bias, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_a_layernorm, kv_lora_rank, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_b_proj, kv_lora_rank, total_num_heads * (qk_nope_head_dim_ + v_head_dim_), quantization_method, false, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * v_head_dim_, hidden_size_, quantization_method, attention_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    softmax_scale_ = deepseek_v2_attention_softmax_scale(model_config, static_cast<float>(q_head_dim_));

    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, q_head_dim_, softmax_scale_, num_attention_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);
    infinilm::layers::attention::init_kv_cache_quant_params(
        [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); },
        device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor DeepseekV2Attention::position_ids_for_rope_(const infinicore::Tensor &position_ids) const {
    auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->contiguous()->view({pos_shape[1]});
    }
    if (pos_shape.size() == 1) {
        return position_ids->contiguous();
    }
    throw std::runtime_error("DeepseekV2Attention: unexpected position_ids shape");
}

infinicore::Tensor DeepseekV2Attention::trim_value_padding_(const infinicore::Tensor &attn_output) const {
    const auto shape = attn_output->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    return attn_output->view({batch_size, seq_len, num_attention_heads_, q_head_dim_})
        ->narrow({{3, 0, v_head_dim_}})
        ->contiguous()
        ->view({batch_size, seq_len, num_attention_heads_ * v_head_dim_});
}

infinicore::Tensor DeepseekV2Attention::forward(const infinicore::Tensor &positions,
                                                const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor DeepseekV2Attention::forward_static_(const infinicore::Tensor &position_ids,
                                                        const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto hidden_states_mutable = hidden_states;

    auto q = q_proj_->forward(hidden_states_mutable)->view({batch_size, seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{3, 0, qk_nope_head_dim_}});
    auto q_pe = q->narrow({{3, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous();

    auto compressed = kv_a_proj_with_mqa_->forward(hidden_states_mutable);
    auto compressed_kv = compressed->narrow({{2, 0, kv_a_layernorm_->normalized_shape()}})->contiguous();
    auto k_pe = compressed->narrow({{2, kv_a_layernorm_->normalized_shape(), qk_rope_head_dim_}})->contiguous();

    auto kv_norm = kv_a_layernorm_->forward(compressed_kv);
    auto kv = kv_b_proj_->forward(kv_norm)->view({batch_size, seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->narrow({{3, 0, qk_nope_head_dim_}});
    auto value_states = kv->narrow({{3, qk_nope_head_dim_, v_head_dim_}})->contiguous();

    auto pos_ids = position_ids_for_rope_(position_ids);
    q_pe = rotary_emb_->forward(q_pe, pos_ids, true);
    auto k_pe_broadcast = infinicore::op::broadcast_to(k_pe->view({batch_size, seq_len, 1, qk_rope_head_dim_}),
                                                       {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len), static_cast<int64_t>(num_attention_heads_), static_cast<int64_t>(qk_rope_head_dim_)});
    k_pe_broadcast = rotary_emb_->forward(k_pe_broadcast, pos_ids, true);

    auto query_states = infinicore::op::cat({q_nope, q_pe}, 3);
    auto key_states = infinicore::op::cat({k_nope, k_pe_broadcast}, 3);
    auto value_padded = infinicore::op::pad(value_states, {0, static_cast<int>(q_head_dim_ - v_head_dim_)}, "constant", 0.0);

    auto attn_output = attn_->forward(query_states, key_states, value_padded);
    auto trimmed_output = trim_value_padding_(attn_output);
    return o_proj_->forward(trimmed_output);
}

infinicore::Tensor DeepseekV2Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                       const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);
    auto hidden_states_mutable = hidden_states;

    auto q = q_proj_->forward(hidden_states_mutable)->view({seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}});
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous();

    auto compressed = kv_a_proj_with_mqa_->forward(hidden_states_mutable)->view({seq_len, kv_a_layernorm_->normalized_shape() + qk_rope_head_dim_});
    auto compressed_kv = compressed->narrow({{1, 0, kv_a_layernorm_->normalized_shape()}})->contiguous();
    auto k_pe = compressed->narrow({{1, kv_a_layernorm_->normalized_shape(), qk_rope_head_dim_}})->contiguous();

    auto kv_norm = kv_a_layernorm_->forward(compressed_kv);
    auto kv = kv_b_proj_->forward(kv_norm)->view({seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->narrow({{2, 0, qk_nope_head_dim_}});
    auto value_states = kv->narrow({{2, qk_nope_head_dim_, v_head_dim_}})->contiguous();

    auto pos_ids = position_ids_for_rope_(position_ids);
    q_pe = rotary_emb_->forward(q_pe, pos_ids, true);
    auto k_pe_broadcast = infinicore::op::broadcast_to(k_pe->view({seq_len, 1, qk_rope_head_dim_}),
                                                       {static_cast<int64_t>(seq_len), static_cast<int64_t>(num_attention_heads_), static_cast<int64_t>(qk_rope_head_dim_)});
    k_pe_broadcast = rotary_emb_->forward(k_pe_broadcast, pos_ids, true);

    auto query_states = infinicore::op::cat({q_nope, q_pe}, 2);
    auto key_states = infinicore::op::cat({k_nope, k_pe_broadcast}, 2);
    auto value_padded = infinicore::op::pad(value_states, {0, static_cast<int>(q_head_dim_ - v_head_dim_)}, "constant", 0.0);

    auto attn_output = attn_->forward(query_states, key_states, value_padded);
    auto trimmed_output = trim_value_padding_(attn_output);
    return o_proj_->forward(trimmed_output);
}

} // namespace infinilm::models::deepseek_v2
