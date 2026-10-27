#include "kimi_k3_mla_attention.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/bmm_strided.hpp>
#include <infinicore/ops/concat_and_cache_mla.hpp>
#include <infinicore/ops/flash_mla.hpp>
#include <infinicore/ops/mha_varlen.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/rearrange.hpp>
#include <infinicore/ops/sigmoid.hpp>

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace infinilm::models::kimi_k3 {
namespace {

infinicore::Tensor build_key(
    const infinicore::Tensor &kv_b,
    const infinicore::Tensor &k_rot,
    size_t num_heads,
    size_t qk_nope_head_dim,
    size_t qk_rope_head_dim) {
    auto key_shape = kv_b->shape();
    const size_t head_axis = key_shape.size() - 2;
    const size_t feature_dim = key_shape.size() - 1;
    key_shape[feature_dim] = qk_nope_head_dim + qk_rope_head_dim;

    auto key = infinicore::Tensor::empty(key_shape, kv_b->dtype(), kv_b->device());
    infinicore::op::rearrange_(
        key->narrow({{feature_dim, 0, qk_nope_head_dim}}),
        kv_b->narrow({{feature_dim, 0, qk_nope_head_dim}}));

    auto k_rot_shape = key_shape;
    k_rot_shape[head_axis] = num_heads;
    k_rot_shape[feature_dim] = qk_rope_head_dim;
    infinicore::Strides k_rot_strides(key_shape.size());
    for (size_t i = 0; i < head_axis; ++i) {
        k_rot_strides[i] = k_rot->stride(i);
    }
    k_rot_strides[head_axis] = 0;
    k_rot_strides[feature_dim] = k_rot->stride(k_rot->ndim() - 1);
    auto k_rot_heads = k_rot->as_strided(k_rot_shape, k_rot_strides);
    infinicore::op::rearrange_(
        key->narrow({{feature_dim, qk_nope_head_dim, qk_rope_head_dim}}),
        k_rot_heads);
    return key;
}

} // namespace

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
    if (attention_backend_ == backends::AttentionBackend::STATIC_ATTN) {
        throw std::runtime_error("KimiK3MLAAttention does not support static attention");
    }
    softmax_scale_ = 1.0f / std::sqrt(static_cast<float>(q_head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        local_num_heads_, q_head_dim_, softmax_scale_, local_num_heads_,
        layer_idx_, kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);
    infinilm::layers::attention::init_kv_cache_quant_params(
        [this](const std::string &name, infinicore::nn::Parameter parameter) {
            this->register_parameter(name, std::move(parameter));
        },
        device,
        kv_cache_k_scale_,
        kv_cache_v_scale_);
    if (!kv_cache_k_scale_) {
        kv_cache_k_scale_ = infinicore::nn::Parameter(
            infinicore::Tensor::ones({1}, infinicore::DataType::F32, device));
    }
}

void KimiK3MLAAttention::process_weights_after_loading() {
    if (std::getenv("INFINILM_KIMI_FLASH_MLA") == nullptr) {
        return;
    }
    const auto weight = kv_b_proj_->weight();
    if (!weight || weight->ndim() != 2 || weight->dtype() != kv_a_layernorm_->dtype()) {
        throw std::runtime_error(
            "KimiK3MLAAttention expects native kv_b_proj weight [out,in]");
    }
    auto weight_3d = weight->view(
        {local_num_heads_, qk_nope_head_dim_ + v_head_dim_, kv_lora_rank_});
    w_uk_ = weight_3d->narrow({{1, 0, qk_nope_head_dim_}})->contiguous();
    w_uv_ = weight_3d->narrow({{1, qk_nope_head_dim_, v_head_dim_}})
                ->permute({0, 2, 1})
                ->contiguous();
}

infinicore::Tensor KimiK3MLAAttention::project_q_nope_to_latent(
    const infinicore::Tensor &q_nope) const {
    if (!w_uk_) {
        throw std::runtime_error("KimiK3MLAAttention absorbed K weight is not prepared");
    }
    auto q_latent = infinicore::Tensor::empty(
        {q_nope->size(0), local_num_heads_, kv_lora_rank_}, q_nope->dtype(),
        q_nope->device());
    infinicore::op::bmm_strided_(q_latent->permute({1, 0, 2}),
                                 q_nope->permute({1, 0, 2}), w_uk_);
    return q_latent;
}

infinicore::Tensor KimiK3MLAAttention::project_latent_to_value(
    const infinicore::Tensor &latent) const {
    if (!w_uv_) {
        throw std::runtime_error("KimiK3MLAAttention absorbed V weight is not prepared");
    }
    auto value = infinicore::Tensor::empty(
        {latent->size(0), local_num_heads_, v_head_dim_}, latent->dtype(),
        latent->device());
    infinicore::op::bmm_strided_(value->permute({1, 0, 2}),
                                 latent->permute({1, 0, 2}), w_uv_);
    return value;
}

infinicore::Tensor KimiK3MLAAttention::forward(
    const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    if (batch_size != 1) {
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

    q = q->view({seq_len, local_num_heads_, q_head_dim_});
    if (std::getenv("INFINILM_KIMI_FLASH_MLA") == nullptr) {
        auto kv = kv_b_proj_->forward(normalized_kv)->view({seq_len, local_num_heads_, qk_nope_head_dim_ + v_head_dim_});
        k_rot = k_rot->squeeze(0);
        auto key = build_key(
            kv, k_rot, local_num_heads_, qk_nope_head_dim_, qk_rope_head_dim_);
        auto value = kv->narrow({{2, qk_nope_head_dim_, v_head_dim_}});
        auto padded_output = attn_->forward(q, key, value)
                                 ->view({batch_size, seq_len, local_num_heads_,
                                         q_head_dim_});
        auto output = infinicore::Tensor::empty(
            {batch_size, seq_len, local_num_heads_, v_head_dim_},
            padded_output->dtype(), padded_output->device());
        auto gate = g_proj_->forward(input)->view(
            {batch_size, seq_len, local_num_heads_, v_head_dim_});
        infinicore::op::sigmoid_(gate, gate);
        infinicore::op::mul_(
            output, padded_output->narrow({{3, 0, v_head_dim_}}), gate);
        auto flattened_output = output->view(
            {batch_size, seq_len, local_num_heads_ * v_head_dim_});
        return o_proj_->forward(flattened_output);
    }
    k_rot = k_rot->squeeze(0)->contiguous();

    auto &metadata = global_state::get_forward_context().attn_metadata;
    auto &kv_cache = global_state::get_forward_context().kv_cache_vec[layer_idx_];
    if (!metadata.slot_mapping.has_value() || !metadata.block_tables.has_value() || !metadata.total_sequence_lengths.has_value()) {
        throw std::runtime_error("KimiK3MLAAttention requires paged attention metadata");
    }
    infinicore::op::concat_and_cache_mla_(
        normalized_kv->view({seq_len, kv_lora_rank_}),
        k_rot->view({seq_len, qk_rope_head_dim_}), kv_cache,
        metadata.slot_mapping.value(), "auto", kv_cache_k_scale_);

    const bool is_prefill = seq_len != metadata.total_sequence_lengths.value()->numel();
    infinicore::Tensor value;
    if (is_prefill) {
        if (attention_backend_ != backends::AttentionBackend::FLASH_ATTN || !metadata.input_offsets.has_value() || !metadata.cu_seqlens.has_value() || !metadata.max_context_len.has_value()) {
            throw std::runtime_error(
                "KimiK3MLAAttention prefill requires flash attention metadata");
        }
        auto kv = kv_b_proj_->forward(normalized_kv)->view({seq_len, local_num_heads_, qk_nope_head_dim_ + v_head_dim_});
        auto key = build_key(
            kv, k_rot, local_num_heads_, qk_nope_head_dim_, qk_rope_head_dim_);
        auto expanded_value = infinicore::Tensor::zeros(
            {seq_len, local_num_heads_, q_head_dim_}, q->dtype(), q->device());
        expanded_value->narrow({{2, 0, v_head_dim_}})
            ->copy_from(kv->narrow({{2, qk_nope_head_dim_, v_head_dim_}}));
        auto padded_output = infinicore::Tensor::empty(
            {seq_len, local_num_heads_, q_head_dim_}, q->dtype(), q->device());
        infinicore::op::mha_varlen_(
            padded_output, q, key, expanded_value,
            metadata.input_offsets.value(),
            metadata.cu_seqlens.value(), std::nullopt,
            static_cast<int>(metadata.max_context_len.value()),
            static_cast<int>(metadata.max_context_len.value()), std::nullopt,
            softmax_scale_);
        value = padded_output->narrow({{2, 0, v_head_dim_}});
    } else {
        auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}});
        auto q_extra = q->narrow(
            {{2, qk_nope_head_dim_, qk_rope_head_dim_}});
        auto q_latent = project_q_nope_to_latent(q_nope);
        auto query = infinicore::Tensor::empty(
            {seq_len, local_num_heads_, kv_lora_rank_ + qk_rope_head_dim_},
            q->dtype(), q->device());
        query->narrow({{2, 0, kv_lora_rank_}})->copy_from(q_latent);
        query->narrow({{2, kv_lora_rank_, qk_rope_head_dim_}})
            ->copy_from(q_extra);
        auto latent = infinicore::Tensor::empty(
            {seq_len, local_num_heads_, kv_lora_rank_}, q->dtype(), q->device());
        if (kv_cache->size(1) != 64) {
            throw std::runtime_error(
                "Kimi K3 FlashMLA decode requires --block-size=64");
        }
        infinicore::op::flash_mla_dense_decode_(
            latent->view({seq_len, 1, local_num_heads_, kv_lora_rank_}),
            query->view({seq_len, 1, local_num_heads_,
                         kv_lora_rank_ + qk_rope_head_dim_}),
            kv_cache->view({kv_cache->size(0), kv_cache->size(1), 1,
                            kv_cache->size(2)}),
            metadata.block_tables.value(),
            metadata.total_sequence_lengths.value(), softmax_scale_);
        value = project_latent_to_value(latent);
    }

    auto value_4d = value->unsqueeze(0);
    auto gate = g_proj_->forward(input)
                    ->view({batch_size, seq_len, local_num_heads_, v_head_dim_});
    infinicore::op::sigmoid_(gate, gate);
    infinicore::Tensor output;
    if (value->is_contiguous()) {
        output = value_4d;
        infinicore::op::mul_(output, output, gate);
    } else {
        output = infinicore::Tensor::empty(
            {batch_size, seq_len, local_num_heads_, v_head_dim_},
            value->dtype(), value->device());
        infinicore::op::mul_(output, value_4d, gate);
    }
    output = output->view({batch_size, seq_len, local_num_heads_ * v_head_dim_});
    return o_proj_->forward(output);
}

} // namespace infinilm::models::kimi_k3
