#include "deepseek_v2_mla_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"
#include "deepseek_v2_utils.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "infinicore/ops/concat_mla_q.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/paged_attention_mla.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v2 {

DeepseekV2MLAAttention::DeepseekV2MLAAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
                                               const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    qk_nope_head_dim_ = model_config->get<size_t>("qk_nope_head_dim");
    qk_rope_head_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    q_head_dim_ = qk_nope_head_dim_ + qk_rope_head_dim_;
    v_head_dim_ = model_config->get<size_t>("v_head_dim");
    kv_lora_rank_ = model_config->get<size_t>("kv_lora_rank");
    mla_head_dim_ = kv_lora_rank_ + qk_rope_head_dim_;

    const auto &config_json = model_config->get_config_json();
    q_lora_rank_ = config_json.contains("q_lora_rank") && !config_json["q_lora_rank"].is_null()
                     ? config_json["q_lora_rank"].get<size_t>()
                     : 0;

    const auto &dtype{model_config->get_dtype()};
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const bool attention_bias = model_config->get_or<bool>("attention_bias", false);
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    if (total_num_heads < static_cast<size_t>(tp_size)
        || total_num_heads % static_cast<size_t>(tp_size) != 0) {
        throw std::runtime_error("DeepseekV2MLAAttention: num_attention_heads must be divisible by tp_size");
    }
    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    if (attention_backend_ == infinilm::backends::AttentionBackend::STATIC_ATTN) {
        throw std::runtime_error("DeepseekV2MLAAttention requires paged or flash attention; the dense MHA path was removed");
    }

    auto quantization_method = model_config->get_quantization_method();
    if (q_lora_rank_ == 0) {
        INFINICORE_NN_MODULE_INIT(q_proj,
                                  hidden_size_,
                                  total_num_heads * q_head_dim_,
                                  quantization_method,
                                  false,
                                  dtype,
                                  device,
                                  tp_rank,
                                  tp_size);
    } else {
        INFINICORE_NN_MODULE_INIT(q_a_proj,
                                  hidden_size_,
                                  q_lora_rank_,
                                  quantization_method,
                                  false,
                                  dtype,
                                  device);
        INFINICORE_NN_MODULE_INIT(q_a_layernorm, q_lora_rank_, rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(q_b_proj,
                                  q_lora_rank_,
                                  total_num_heads * q_head_dim_,
                                  quantization_method,
                                  false,
                                  dtype,
                                  device,
                                  tp_rank,
                                  tp_size);
    }
    INFINICORE_NN_MODULE_INIT(kv_a_proj_with_mqa,
                              hidden_size_,
                              kv_lora_rank_ + qk_rope_head_dim_,
                              quantization_method,
                              attention_bias,
                              dtype,
                              device);
    INFINICORE_NN_MODULE_INIT(kv_a_layernorm, kv_lora_rank_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_b_proj,
                              kv_lora_rank_,
                              total_num_heads * (qk_nope_head_dim_ + v_head_dim_),
                              quantization_method,
                              false,
                              dtype,
                              device,
                              tp_rank,
                              tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj,
                              total_num_heads * v_head_dim_,
                              hidden_size_,
                              quantization_method,
                              attention_bias,
                              dtype,
                              device,
                              tp_rank,
                              tp_size,
                              rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    softmax_scale_ = deepseek_v2_attention_softmax_scale(model_config, static_cast<float>(q_head_dim_));
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

infinicore::Tensor DeepseekV2MLAAttention::position_ids_for_rope_(const infinicore::Tensor &position_ids) const {
    const auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->contiguous()->view({pos_shape[1]});
    }
    if (pos_shape.size() == 1) {
        return position_ids->contiguous();
    }
    throw std::runtime_error("DeepseekV2MLAAttention: unexpected position_ids shape");
}

infinicore::Tensor DeepseekV2MLAAttention::kv_b_weight_3d_() const {
    return kv_b_proj_->weight()->view(
        {num_attention_heads_, qk_nope_head_dim_ + v_head_dim_, kv_lora_rank_});
}

infinicore::Tensor DeepseekV2MLAAttention::project_q_nope_to_latent_(const infinicore::Tensor &q_nope) const {
    const size_t num_tokens = q_nope->shape()[0];
    auto q_nope_by_head = q_nope->permute({1, 0, 2})->contiguous();
    auto w_uk_t = kv_b_weight_3d_()->narrow({{1, 0, qk_nope_head_dim_}})->contiguous();
    auto q_latent = infinicore::op::matmul(q_nope_by_head, w_uk_t);
    return q_latent->permute({1, 0, 2})
        ->contiguous()
        ->view({num_tokens, num_attention_heads_, kv_lora_rank_});
}

infinicore::Tensor DeepseekV2MLAAttention::project_latent_to_value_(const infinicore::Tensor &attn_output,
                                                                    size_t batch_size,
                                                                    size_t seq_len) const {
    const size_t num_tokens = batch_size * seq_len;
    auto latent_by_head = attn_output->view({num_tokens, num_attention_heads_, kv_lora_rank_})
                              ->permute({1, 0, 2})
                              ->contiguous();
    auto w_uv = kv_b_weight_3d_()
                    ->narrow({{1, qk_nope_head_dim_, v_head_dim_}})
                    ->permute({0, 2, 1})
                    ->contiguous();
    auto value = infinicore::op::matmul(latent_by_head, w_uv)
                     ->permute({1, 0, 2})
                     ->contiguous()
                     ->view({batch_size, seq_len, num_attention_heads_ * v_head_dim_});
    return o_proj_->forward(value);
}

infinicore::Tensor DeepseekV2MLAAttention::forward(const infinicore::Tensor &position_ids,
                                                   const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    if (batch_size != 1) {
        throw std::runtime_error("DeepseekV2MLAAttention currently expects batch_size=1 in the paged engine");
    }

    auto hidden_states_mutable = hidden_states;
    infinicore::Tensor q_linear;
    if (q_lora_rank_ == 0) {
        q_linear = q_proj_->forward(hidden_states_mutable);
    } else {
        auto q_a = q_a_proj_->forward(hidden_states_mutable);
        auto q_a_norm = q_a_layernorm_->forward(q_a);
        q_linear = q_b_proj_->forward(q_a_norm);
    }
    auto q = q_linear->view({seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}})->contiguous();
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous();

    auto compressed = kv_a_proj_with_mqa_->forward(hidden_states_mutable)
                          ->view({seq_len, kv_lora_rank_ + qk_rope_head_dim_});
    auto compressed_kv = compressed->narrow({{1, 0, kv_lora_rank_}})->contiguous();
    auto k_pe = compressed->narrow({{1, kv_lora_rank_, qk_rope_head_dim_}})->contiguous();
    auto kv_norm = kv_a_layernorm_->forward(compressed_kv);

    auto pos_ids = position_ids_for_rope_(position_ids);
    q_pe = rotary_emb_->forward(q_pe, pos_ids, true);
    auto k_pe_rope = rotary_emb_->forward(
        k_pe->view({seq_len, 1, qk_rope_head_dim_}), pos_ids, true);

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto cu_seqlens = attn_metadata.cu_seqlens;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());
    ASSERT(total_sequence_lengths.has_value());
    ASSERT(attn_metadata.max_context_len.has_value());

    if (hidden_states->device().getType() != infinicore::Device::Type::ILUVATAR) {
        throw std::runtime_error("DeepseekV2MLAAttention: the vLLM-style MLA cache path currently requires Iluvatar");
    }
    infinicore::op::concat_and_cache_mla_(kv_norm,
                                          k_pe_rope->view({seq_len, qk_rope_head_dim_}),
                                          kv_cache,
                                          slot_mapping.value(),
                                          "auto",
                                          kv_cache_k_scale_);

    const bool is_prefill = seq_len != total_sequence_lengths.value()->shape()[0];
    if (is_prefill) {
        if (attention_backend_ != infinilm::backends::AttentionBackend::FLASH_ATTN) {
            throw std::runtime_error("DeepseekV2MLAAttention: prefill requires --attn=flash-attn");
        }
        ASSERT(input_offsets.has_value());
        ASSERT(cu_seqlens.has_value());
        const size_t num_requests = total_sequence_lengths.value()->shape()[0];
        ASSERT(num_requests > 0);
        ASSERT_EQ(seq_len % num_requests, 0);
        const int max_seqlen = static_cast<int>(seq_len / num_requests);

        auto kv_b = kv_b_proj_->forward(kv_norm)->view(
            {seq_len, num_attention_heads_, qk_nope_head_dim_ + v_head_dim_});
        auto key_nope = kv_b->narrow({{2, 0, qk_nope_head_dim_}})->contiguous();
        auto value_states = kv_b->narrow({{2, qk_nope_head_dim_, v_head_dim_}})->contiguous();
        auto key_pe = infinicore::op::broadcast_to(
                          k_pe_rope,
                          {static_cast<int64_t>(seq_len),
                           static_cast<int64_t>(num_attention_heads_),
                           static_cast<int64_t>(qk_rope_head_dim_)})
                          ->contiguous();
        auto query_states = infinicore::op::cat({q_nope, q_pe}, 2);
        auto key_states = infinicore::op::cat({key_nope, key_pe}, 2);
        auto attn_output = infinicore::Tensor::empty(
            {seq_len, num_attention_heads_, v_head_dim_}, query_states->dtype(), query_states->device());
        infinicore::op::mha_varlen_(attn_output,
                                    query_states,
                                    key_states,
                                    value_states,
                                    input_offsets.value(),
                                    cu_seqlens.value(),
                                    std::nullopt,
                                    max_seqlen,
                                    max_seqlen,
                                    std::nullopt,
                                    softmax_scale_);
        auto projected = attn_output->view(
            {batch_size, seq_len, num_attention_heads_ * v_head_dim_});
        return o_proj_->forward(projected);
    }

    auto q_latent = project_q_nope_to_latent_(q_nope);
    auto query_states = infinicore::op::concat_mla_q(q_latent, q_pe);
    auto attn_output = infinicore::Tensor::empty(
        {seq_len, num_attention_heads_, kv_lora_rank_}, query_states->dtype(), query_states->device());
    const int64_t max_context_len = attn_metadata.max_context_len.value();
    infinicore::op::paged_attention_mla_(attn_output,
                                         query_states,
                                         kv_cache,
                                         softmax_scale_,
                                         block_tables.value(),
                                         total_sequence_lengths.value(),
                                         max_context_len);
    return project_latent_to_value_(attn_output, batch_size, seq_len);
}

} // namespace infinilm::models::deepseek_v2
