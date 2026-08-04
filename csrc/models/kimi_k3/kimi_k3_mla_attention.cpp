#include "kimi_k3_mla_attention.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/broadcast_to.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/pad.hpp>
#include <infinicore/ops/sigmoid.hpp>

#include <cmath>
#include <stdexcept>

namespace infinilm::models::kimi_k3 {

KimiK3MLAAttention::KimiK3MLAAttention(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    q_lora_rank_ = model_config->get<size_t>("q_lora_rank");
    kv_lora_rank_ = model_config->get<size_t>("kv_lora_rank");
    qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    v_head_dim_ = model_config->get<size_t>("v_head_dim");
    const size_t num_heads = model_config->get<size_t>("num_attention_heads");
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    if (!model_config->get<bool>("mla_use_nope")) {
        throw std::runtime_error("KimiK3MLAAttention only supports K3's NoPE MLA");
    }
    if (num_heads % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error("KimiK3MLAAttention: num_attention_heads must be divisible by tp_size");
    }
    local_num_heads_ = num_heads / static_cast<size_t>(rank_info.tp_size);
    const auto &dtype = model_config->get_dtype();
    const double eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(q_a_proj, hidden_size, q_lora_rank_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(q_a_layernorm, q_lora_rank_, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(q_b_proj, q_lora_rank_, num_heads * q_head_dim_, false,
                              dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(kv_a_proj_with_mqa, hidden_size,
                              kv_lora_rank_ + qk_rope_head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_a_layernorm, kv_lora_rank_, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_b_proj, kv_lora_rank_,
                              num_heads * (qk_nope_head_dim_ + v_head_dim_), false,
                              dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(g_proj, hidden_size, num_heads * v_head_dim_, false,
                              dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj, num_heads * v_head_dim_, hidden_size, false,
                              dtype, device, rank_info.tp_rank, rank_info.tp_size,
                              rank_info.comm);

    attention_backend_ = global_state::get_infinilm_config().attention_backend;
    softmax_scale_ = 1.0f / std::sqrt(static_cast<float>(q_head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        local_num_heads_, q_head_dim_, softmax_scale_, local_num_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);
    infinilm::layers::attention::init_kv_cache_quant_params(
        [this](const std::string &name, infinicore::nn::Parameter parameter) {
            this->register_parameter(name, std::move(parameter));
        },
        device,
        kv_cache_k_scale_,
        kv_cache_v_scale_);
}

infinicore::Tensor KimiK3MLAAttention::trim_value_padding(
    const infinicore::Tensor &output) const {
    const auto shape = output->shape();
    return output->view({shape[0], shape[1], local_num_heads_, q_head_dim_})
        ->narrow({{3, 0, v_head_dim_}})
        ->contiguous()
        ->view({shape[0], shape[1], local_num_heads_ * v_head_dim_});
}

infinicore::Tensor KimiK3MLAAttention::forward(
    const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    if (attention_backend_ != backends::AttentionBackend::STATIC_ATTN && batch_size != 1) {
        throw std::runtime_error("KimiK3MLAAttention: paged attention expects flattened batch size 1");
    }
    auto input = hidden_states;
    auto q_lora = q_a_proj_->forward(input);
    q_lora = q_a_layernorm_->forward(q_lora);
    auto q = q_b_proj_->forward(q_lora);
    auto compressed = kv_a_proj_with_mqa_->forward(input);
    auto compressed_kv = compressed->narrow({{2, 0, kv_lora_rank_}});
    auto k_rot = compressed->narrow({{2, kv_lora_rank_, qk_rope_head_dim_}});
    auto normalized_kv = kv_a_layernorm_->forward(compressed_kv);
    auto kv = kv_b_proj_->forward(normalized_kv);

    if (attention_backend_ == backends::AttentionBackend::STATIC_ATTN) {
        q = q->view({batch_size, seq_len, local_num_heads_, q_head_dim_});
        kv = kv->view({batch_size, seq_len, local_num_heads_, qk_nope_head_dim_ + v_head_dim_});
        auto k_nope = kv->narrow({{3, 0, qk_nope_head_dim_}});
        auto value = kv->narrow({{3, qk_nope_head_dim_, v_head_dim_}});
        auto k_rot_heads = infinicore::op::broadcast_to(
            k_rot->view({batch_size, seq_len, 1, qk_rope_head_dim_}),
            {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len),
             static_cast<int64_t>(local_num_heads_), static_cast<int64_t>(qk_rope_head_dim_)});
        auto key = infinicore::op::cat({k_nope, k_rot_heads}, 3);
        auto value_padded = infinicore::op::pad(
            value, {0, static_cast<int>(q_head_dim_ - v_head_dim_)}, "constant", 0.0);
        auto output = trim_value_padding(attn_->forward(q, key, value_padded));
        output = infinicore::op::mul(output, infinicore::op::sigmoid(g_proj_->forward(input)));
        return o_proj_->forward(output);
    }

    q = q->view({seq_len, local_num_heads_, q_head_dim_});
    kv = kv->view({seq_len, local_num_heads_, qk_nope_head_dim_ + v_head_dim_});
    auto k_nope = kv->narrow({{2, 0, qk_nope_head_dim_}});
    auto value = kv->narrow({{2, qk_nope_head_dim_, v_head_dim_}});
    auto k_rot_heads = infinicore::op::broadcast_to(
        k_rot->view({seq_len, 1, qk_rope_head_dim_}),
        {static_cast<int64_t>(seq_len), static_cast<int64_t>(local_num_heads_),
         static_cast<int64_t>(qk_rope_head_dim_)});
    auto key = infinicore::op::cat({k_nope, k_rot_heads}, 2);
    auto value_padded = infinicore::op::pad(
        value, {0, static_cast<int>(q_head_dim_ - v_head_dim_)}, "constant", 0.0);
    auto output = trim_value_padding(attn_->forward(q, key, value_padded));
    output = infinicore::op::mul(output, infinicore::op::sigmoid(g_proj_->forward(input)));
    return o_proj_->forward(output);
}

} // namespace infinilm::models::kimi_k3
