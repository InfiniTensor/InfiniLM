#include "deepseek_v2_mla_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/pad.hpp"
#include "infinicore/ops/paged_attention.hpp"
#include "infinicore/ops/paged_attention_prefill.hpp"
#include "infinicore/ops/paged_caching.hpp"

#include <atomic>
#include <cmath>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v2 {
namespace {

std::atomic<int> g_dense_mla_prefill_state{0}; // 0 unknown, 1 available, 2 unavailable

bool is_dense_mla_prefill_unavailable(const std::runtime_error &err) {
    const std::string msg = err.what();
    return msg.find("ATen is not enabled") != std::string::npos
        || msg.find("FlashAttention is not enabled") != std::string::npos;
}

float yarn_get_mscale(float scale, float mscale) {
    if (scale <= 1.0f) {
        return 1.0f;
    }
    return 0.1f * mscale * std::log(scale) + 1.0f;
}

} // namespace

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

    if (model_config->get_or<size_t>("q_lora_rank", 0) != 0) {
        throw std::runtime_error("DeepseekV2MLAAttention: q_lora_rank is not supported yet");
    }

    const auto &dtype{model_config->get_dtype()};
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const bool attention_bias = model_config->get_or<bool>("attention_bias", false);
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    if ((total_num_heads < static_cast<size_t>(tp_size)) || (total_num_heads % static_cast<size_t>(tp_size) != 0)) {
        throw std::runtime_error("DeepseekV2MLAAttention: num_attention_heads must be divisible by tp_size");
    }
    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;

    auto quantization_method = model_config->get_quantization_method();
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, total_num_heads * q_head_dim_, quantization_method, false, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(kv_a_proj_with_mqa, hidden_size_, kv_lora_rank_ + qk_rope_head_dim_, attention_bias, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_a_layernorm, kv_lora_rank_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_b_proj, kv_lora_rank_, total_num_heads * (qk_nope_head_dim_ + v_head_dim_), quantization_method, false, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * v_head_dim_, hidden_size_, quantization_method, attention_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    const double rope_theta = model_config->get<double>("rope_theta");
    rotary_emb_ = std::make_shared<infinicore::nn::RoPE>(
        qk_rope_head_dim_, qk_rope_head_dim_, max_position_embeddings, rope_theta,
        infinicore::nn::RoPE::Algo::GPT_J, dtype, device, nullptr);

    softmax_scale_ = 1.0f / std::sqrt(static_cast<float>(q_head_dim_));
    auto &config_json = model_config->get_config_json();
    if (config_json.contains("rope_scaling") && config_json["rope_scaling"].is_object()) {
        const auto &rope_scaling = config_json["rope_scaling"];
        const float mscale_all_dim = rope_scaling.value("mscale_all_dim", 0.0f);
        if (mscale_all_dim != 0.0f) {
            const float scaling_factor = rope_scaling.value("factor", 1.0f);
            const float mscale = yarn_get_mscale(scaling_factor, mscale_all_dim);
            softmax_scale_ *= mscale * mscale;
        }
    }

    latent_attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, mla_head_dim_, softmax_scale_, 1, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);
    infinilm::layers::attention::init_kv_cache_quant_params(
        [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); },
        device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor DeepseekV2MLAAttention::position_ids_for_rope_(const infinicore::Tensor &position_ids) const {
    auto pos_shape = position_ids->shape();
    if (pos_shape.size() == 2) {
        return position_ids->narrow({{0, 0, 1}})->contiguous()->view({pos_shape[1]});
    }
    if (pos_shape.size() == 1) {
        return position_ids->contiguous();
    }
    throw std::runtime_error("DeepseekV2MLAAttention: unexpected position_ids shape");
}

infinicore::Tensor DeepseekV2MLAAttention::kv_b_weight_3d_() const {
    return kv_b_proj_->weight()->view({num_attention_heads_, qk_nope_head_dim_ + v_head_dim_, kv_lora_rank_});
}

infinicore::Tensor DeepseekV2MLAAttention::project_q_nope_to_latent_(const infinicore::Tensor &q_nope) const {
    const size_t ntokens = q_nope->shape()[0];
    auto q_nope_by_head = q_nope->permute({1, 0, 2})->contiguous();
    auto w_uk_t = kv_b_weight_3d_()->narrow({{1, 0, qk_nope_head_dim_}})->contiguous();
    auto q_latent = infinicore::op::matmul(q_nope_by_head, w_uk_t);
    return q_latent->permute({1, 0, 2})->contiguous()->view({ntokens, num_attention_heads_, kv_lora_rank_});
}

infinicore::Tensor DeepseekV2MLAAttention::project_latent_to_value_(const infinicore::Tensor &attn_output,
                                                                    size_t batch_size,
                                                                    size_t seq_len) const {
    const size_t ntokens = batch_size * seq_len;
    const auto out_shape = attn_output->shape();
    const size_t out_head_dim = out_shape.back();
    infinicore::Tensor latent;
    if (out_head_dim == kv_lora_rank_) {
        latent = attn_output->view({ntokens, num_attention_heads_, kv_lora_rank_});
    } else {
        latent = attn_output->view({ntokens, num_attention_heads_, mla_head_dim_})
                     ->narrow({{2, 0, kv_lora_rank_}})
                     ->contiguous();
    }
    auto latent_by_head = latent->permute({1, 0, 2})->contiguous();
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

infinicore::Tensor DeepseekV2MLAAttention::forward(const infinicore::Tensor &positions,
                                                   const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor DeepseekV2MLAAttention::forward_static_(const infinicore::Tensor &position_ids,
                                                           const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t ntokens = batch_size * seq_len;
    auto hidden_states_mutable = hidden_states;

    auto q = q_proj_->forward(hidden_states_mutable)->view({ntokens, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}})->contiguous();
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous();

    auto compressed = kv_a_proj_with_mqa_->forward(hidden_states_mutable)->view({ntokens, kv_lora_rank_ + qk_rope_head_dim_});
    auto compressed_kv = compressed->narrow({{1, 0, kv_lora_rank_}})->contiguous();
    auto k_pe = compressed->narrow({{1, kv_lora_rank_, qk_rope_head_dim_}})->contiguous();

    auto kv_norm = kv_a_layernorm_->forward(compressed_kv);
    auto pos_ids = position_ids_for_rope_(position_ids);
    q_pe = rotary_emb_->forward(q_pe, pos_ids, true);
    auto k_pe_rope = rotary_emb_->forward(k_pe->view({ntokens, 1, qk_rope_head_dim_}), pos_ids, true);

    auto q_latent = project_q_nope_to_latent_(q_nope);
    auto query_states = infinicore::op::cat({q_latent, q_pe}, 2)->view({batch_size, seq_len, num_attention_heads_, mla_head_dim_});
    auto key_states = infinicore::op::cat({kv_norm->view({ntokens, 1, kv_lora_rank_}), k_pe_rope}, 2)
                          ->view({batch_size, seq_len, 1, mla_head_dim_});
    auto value_states = infinicore::op::pad(kv_norm->view({batch_size, seq_len, 1, kv_lora_rank_}),
                                            {0, static_cast<int>(qk_rope_head_dim_)}, "constant", 0.0);

    auto attn_output = latent_attn_->forward(query_states, key_states, value_states);
    return project_latent_to_value_(attn_output, batch_size, seq_len);
}

infinicore::Tensor DeepseekV2MLAAttention::forward_paged_(const infinicore::Tensor &position_ids,
                                                          const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);
    auto hidden_states_mutable = hidden_states;

    auto q = q_proj_->forward(hidden_states_mutable)->view({seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}})->contiguous();
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous();

    auto compressed = kv_a_proj_with_mqa_->forward(hidden_states_mutable)->view({seq_len, kv_lora_rank_ + qk_rope_head_dim_});
    auto compressed_kv = compressed->narrow({{1, 0, kv_lora_rank_}})->contiguous();
    auto k_pe = compressed->narrow({{1, kv_lora_rank_, qk_rope_head_dim_}})->contiguous();

    auto kv_norm = kv_a_layernorm_->forward(compressed_kv);
    auto pos_ids = position_ids_for_rope_(position_ids);
    q_pe = rotary_emb_->forward(q_pe, pos_ids, true);
    auto k_pe_rope = rotary_emb_->forward(k_pe->view({seq_len, 1, qk_rope_head_dim_}), pos_ids, true);

    auto q_latent = project_q_nope_to_latent_(q_nope);
    auto query_states = infinicore::op::cat({q_latent, q_pe}, 2);
    auto key_states = infinicore::op::cat({kv_norm->view({seq_len, 1, kv_lora_rank_}), k_pe_rope}, 2);
    auto value_states = kv_norm->view({seq_len, 1, kv_lora_rank_});

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

    auto k_cache_layer = kv_cache->narrow({{3, 0, mla_head_dim_}});
    auto v_cache_layer = kv_cache->narrow({{3, mla_head_dim_, kv_lora_rank_}});
    infinicore::op::paged_caching_(k_cache_layer, v_cache_layer, key_states, value_states, slot_mapping.value());

    auto attn_output = infinicore::Tensor::empty({seq_len, num_attention_heads_, kv_lora_rank_}, query_states->dtype(), query_states->device());
    const bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);
    if (is_prefill) {
        ASSERT(input_offsets.has_value());
        ASSERT(cu_seqlens.has_value());
        const size_t num_reqs = total_sequence_lengths.value()->shape()[0];
        const bool dense_equal_length_prefill = (num_reqs > 0) && (seq_len % num_reqs == 0);
        bool used_dense_prefill = false;
        if (dense_equal_length_prefill && g_dense_mla_prefill_state.load(std::memory_order_relaxed) != 2) {
            const int max_seqlen = static_cast<int>(seq_len / num_reqs);
            try {
                infinicore::op::mha_varlen_(
                    attn_output,
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
                g_dense_mla_prefill_state.store(1, std::memory_order_relaxed);
                used_dense_prefill = true;
            } catch (const std::runtime_error &err) {
                if (!is_dense_mla_prefill_unavailable(err)) {
                    throw;
                }
                g_dense_mla_prefill_state.store(2, std::memory_order_relaxed);
            }
        }
        if (!used_dense_prefill) {
            infinicore::op::paged_attention_prefill_(
                attn_output,
                query_states,
                k_cache_layer,
                v_cache_layer,
                block_tables.value(),
                total_sequence_lengths.value(),
                input_offsets.value(),
                std::nullopt,
                softmax_scale_);
        }
    } else {
        infinicore::op::paged_attention_(
            attn_output,
            query_states,
            k_cache_layer,
            v_cache_layer,
            block_tables.value(),
            total_sequence_lengths.value(),
            std::nullopt,
            softmax_scale_);
    }
    return project_latent_to_value_(attn_output, batch_size, seq_len);
}

} // namespace infinilm::models::deepseek_v2
