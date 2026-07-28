#include "kimi_k25_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../deepseek_v2/deepseek_v2_utils.hpp"

#include <infinicore/ops/broadcast_to.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/pad.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>

namespace infinilm::models::kimi_k25 {

KimiK25Attention::KimiK25Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t layer_idx,
                                   const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    hidden_size_ = model_config->get<size_t>("hidden_size");
    q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
    kv_lora_rank_ = model_config->get<size_t>("kv_lora_rank");
    qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    v_head_dim_ = model_config->get<size_t>("v_head_dim");

    const auto &dtype = model_config->get_dtype();
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const bool attention_bias = model_config->get_or<bool>("attention_bias", false);
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    if (total_num_heads < tp_size || total_num_heads % tp_size != 0) {
        throw std::runtime_error("KimiK25Attention: num_attention_heads must be divisible by tp_size");
    }
    num_attention_heads_ = total_num_heads / tp_size;
    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;

    INFINICORE_NN_MODULE_INIT(q_a_proj, hidden_size_, q_lora_rank_, attention_bias, dtype, device);
    INFINICORE_NN_MODULE_INIT(q_a_layernorm, q_lora_rank_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(q_b_proj, q_lora_rank_, total_num_heads * q_head_dim_, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(kv_a_proj_with_mqa, hidden_size_, kv_lora_rank_ + qk_rope_head_dim_, attention_bias, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_a_layernorm, kv_lora_rank_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_b_proj, kv_lora_rank_, total_num_heads * (qk_nope_head_dim_ + v_head_dim_), false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * v_head_dim_, hidden_size_, attention_bias, dtype, device, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    softmax_scale_ = deepseek_v2::deepseek_v2_attention_softmax_scale(
        model_config, static_cast<float>(q_head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, q_head_dim_, softmax_scale_, num_attention_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);
    infinilm::layers::attention::init_kv_cache_quant_params(
        [this](const std::string &name, infinicore::nn::Parameter parameter) {
            this->register_parameter(name, std::move(parameter));
        },
        device,
        kv_cache_k_scale_,
        kv_cache_v_scale_);
}

infinicore::Tensor KimiK25Attention::position_ids_for_rope(const infinicore::Tensor &position_ids) const {
    const auto shape = position_ids->shape();
    if (shape.size() == 1) {
        return position_ids;
    }
    if (shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->view({shape[1]});
    }
    throw std::runtime_error("KimiK25Attention: unexpected position_ids shape");
}

infinicore::Tensor KimiK25Attention::trim_value_padding(const infinicore::Tensor &attn_output) const {
    const auto shape = attn_output->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    return attn_output->view({batch_size, seq_len, num_attention_heads_, q_head_dim_})
        ->narrow({{3, 0, v_head_dim_}})
        ->contiguous()
        ->view({batch_size, seq_len, num_attention_heads_ * v_head_dim_});
}

infinicore::Tensor KimiK25Attention::forward(const infinicore::Tensor &positions,
                                             const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    if (attention_backend_ != backends::AttentionBackend::STATIC_ATTN && batch_size != 1) {
        throw std::runtime_error("KimiK25Attention: paged attention expects flattened batch size 1");
    }

    auto hidden_mut = hidden_states;
    auto q_lora = q_a_proj_->forward(hidden_mut);
    q_lora = q_a_layernorm_->forward(q_lora);
    auto q = q_b_proj_->forward(q_lora);
    auto compressed = kv_a_proj_with_mqa_->forward(hidden_mut);
    auto compressed_kv = compressed->narrow({{2, 0, kv_lora_rank_}})->contiguous();
    auto k_pe = compressed->narrow({{2, kv_lora_rank_, qk_rope_head_dim_}})->contiguous();
    auto kv_norm = kv_a_layernorm_->forward(compressed_kv);
    auto kv = kv_b_proj_->forward(kv_norm);
    const auto pos_ids = position_ids_for_rope(positions);

    if (attention_backend_ == backends::AttentionBackend::STATIC_ATTN) {
        q = q->view({batch_size, seq_len, num_attention_heads_, q_head_dim_});
        auto q_nope = q->narrow({{3, 0, qk_nope_head_dim_}});
        auto q_pe = q->narrow({{3, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous();
        kv = kv->view({batch_size, seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
        auto k_nope = kv->narrow({{3, 0, qk_nope_head_dim_}});
        auto value = kv->narrow({{3, qk_nope_head_dim_, v_head_dim_}})->contiguous();

        rotary_emb_->forward(q_pe, pos_ids, true);
        auto k_pe_heads = infinicore::op::broadcast_to(
            k_pe->view({batch_size, seq_len, 1, qk_rope_head_dim_}),
            {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len),
             static_cast<int64_t>(num_attention_heads_), static_cast<int64_t>(qk_rope_head_dim_)});
        rotary_emb_->forward(k_pe_heads, pos_ids, true);
        auto query = infinicore::op::cat({q_nope, q_pe}, 3);
        auto key = infinicore::op::cat({k_nope, k_pe_heads}, 3);
        auto value_padded = infinicore::op::pad(value, {0, static_cast<int>(q_head_dim_ - v_head_dim_)}, "constant", 0.0);
        auto output = trim_value_padding(attn_->forward(query, key, value_padded));
        return o_proj_->forward(output);
    }

    q = q->view({seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}});
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous();
    kv = kv->view({seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->narrow({{2, 0, qk_nope_head_dim_}});
    auto value = kv->narrow({{2, qk_nope_head_dim_, v_head_dim_}})->contiguous();

    rotary_emb_->forward(q_pe, pos_ids, true);
    auto k_pe_heads = infinicore::op::broadcast_to(
        k_pe->view({seq_len, 1, qk_rope_head_dim_}),
        {static_cast<int64_t>(seq_len), static_cast<int64_t>(num_attention_heads_),
         static_cast<int64_t>(qk_rope_head_dim_)});
    rotary_emb_->forward(k_pe_heads, pos_ids, true);
    auto query = infinicore::op::cat({q_nope, q_pe}, 2);
    auto key = infinicore::op::cat({k_nope, k_pe_heads}, 2);
    auto value_padded = infinicore::op::pad(value, {0, static_cast<int>(q_head_dim_ - v_head_dim_)}, "constant", 0.0);
    auto output = trim_value_padding(attn_->forward(query, key, value_padded));
    return o_proj_->forward(output);
}

} // namespace infinilm::models::kimi_k25
