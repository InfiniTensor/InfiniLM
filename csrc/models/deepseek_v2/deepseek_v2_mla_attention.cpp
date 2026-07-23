#include "deepseek_v2_mla_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"
#include "deepseek_v2_utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/bmm_strided.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "infinicore/ops/concat_mla_q.hpp"
#include "infinicore/ops/dsa.hpp"
#include "infinicore/ops/fp8_mla_rmsnorm_cache.hpp"
#include "infinicore/ops/fused_rotary_embedding.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/paged_attention_mla.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v2 {
namespace {
void debug_dump_dsa(const infinicore::Tensor &tensor, const std::string &name, size_t layer_idx) {
    if (layer_idx != 0 || std::getenv("INFINILM_GLM_DEBUG_DUMP") == nullptr) {
        return;
    }
    const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank.tp_rank == 0) {
        tensor->debug("/tmp/glmdbg_dsa_" + name + ".bin");
    }
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
    use_sparse_ = config_json.contains("index_topk");
    if (use_sparse_) {
        if (q_lora_rank_ == 0) {
            throw std::runtime_error("Sparse MLA requires q_lora_rank");
        }
        const auto indexer_types = config_json.value("indexer_types", nlohmann::json::array());
        if (layer_idx_ < indexer_types.size()) {
            skip_topk_ = indexer_types[layer_idx_].get<std::string>() == "shared";
        }
        if (!skip_topk_) {
            INFINICORE_NN_MODULE_INIT(indexer, model_config, layer_idx_, device);
        }
    }
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
        fused_qkv_a_proj_ = std::make_shared<infinilm::layers::linear::MergedReplicatedLinear>(
            hidden_size_,
            std::vector<size_t>{q_lora_rank_, kv_lora_rank_ + qk_rope_head_dim_},
            std::vector<std::string>{"q_a_proj", "kv_a_proj_with_mqa"},
            [this](const std::string &name, infinicore::nn::Parameter parameter) {
                this->register_parameter(name, std::move(parameter));
            },
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
    if (q_lora_rank_ == 0) {
        INFINICORE_NN_MODULE_INIT(kv_a_proj_with_mqa,
                                  hidden_size_,
                                  kv_lora_rank_ + qk_rope_head_dim_,
                                  quantization_method,
                                  attention_bias,
                                  dtype,
                                  device);
    }
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
    rope_cos_sin_cache_ = infinilm::layers::rotary_embedding::get_rope_cos_sin_cache(rotary_emb_);
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

void DeepseekV2MLAAttention::process_weights_after_loading() {
    if (fused_qkv_a_proj_) {
        fused_qkv_a_proj_->process_weights_after_loading();
    }
    const auto weight = kv_b_proj_->weight();
    const auto weight_scale = kv_b_proj_->weight_scale();
    if (!weight || weight->dtype() != infinicore::DataType::I8
        || weight->ndim() != 2) {
        throw std::runtime_error(
            "DeepseekV2MLAAttention expects kv_b_proj int8 weight runtime [in,out]");
    }
    if (!weight_scale || weight_scale->dtype() != infinicore::DataType::F32
        || weight_scale->ndim() != 2
        || weight_scale->size(0) != weight->size(1)
        || weight_scale->size(1) != 1) {
        throw std::runtime_error(
            "DeepseekV2MLAAttention expects kv_b_proj float32 weight_scale [out,1]");
    }

    auto out_in_weight = weight->permute({1, 0})->contiguous();
    auto weight_f32 = infinicore::Tensor::empty(
        out_in_weight->shape(), infinicore::DataType::F32, out_in_weight->device());
    infinicore::op::cast_(weight_f32, out_in_weight);
    auto expanded_scale = infinicore::op::broadcast_to(
        weight_scale, {static_cast<int64_t>(out_in_weight->size(0)),
                       static_cast<int64_t>(out_in_weight->size(1))});
    auto dequantized_f32 = infinicore::op::mul(weight_f32, expanded_scale);
    auto dequantized = infinicore::Tensor::empty(
        out_in_weight->shape(), kv_a_layernorm_->dtype(), out_in_weight->device());
    infinicore::op::cast_(dequantized, dequantized_f32);

    auto weight_3d = dequantized->view(
        {num_attention_heads_, qk_nope_head_dim_ + v_head_dim_, kv_lora_rank_});
    w_uk_ = weight_3d->narrow({{1, 0, qk_nope_head_dim_}})->contiguous();
    w_uv_ = weight_3d->narrow({{1, qk_nope_head_dim_, v_head_dim_}})
                ->permute({0, 2, 1})
                ->contiguous();
    kv_b_proj_->release_parameters();
}

void DeepseekV2MLAAttention::reset_runtime_state() const {
    if (fused_qkv_a_proj_) {
        fused_qkv_a_proj_->reset_runtime_state();
    }
}

infinicore::Tensor DeepseekV2MLAAttention::project_q_nope_to_latent_(const infinicore::Tensor &q_nope) const {
    if (!w_uk_) {
        throw std::runtime_error("DeepseekV2MLAAttention absorbed query weight is not prepared");
    }
    const size_t num_tokens = q_nope->shape()[0];
    auto q_latent = infinicore::Tensor::empty(
        {num_tokens, num_attention_heads_, kv_lora_rank_},
        q_nope->dtype(),
        q_nope->device());
    infinicore::op::bmm_strided_(
        q_latent->permute({1, 0, 2}),
        q_nope->permute({1, 0, 2}),
        w_uk_);
    return q_latent;
}

infinicore::Tensor DeepseekV2MLAAttention::project_latent_to_value_(const infinicore::Tensor &attn_output,
                                                                    size_t batch_size,
                                                                    size_t seq_len) const {
    if (!w_uv_) {
        throw std::runtime_error("DeepseekV2MLAAttention absorbed value weight is not prepared");
    }
    const size_t num_tokens = batch_size * seq_len;
    auto latent_by_head = attn_output->view({num_tokens, num_attention_heads_, kv_lora_rank_})
                              ->permute({1, 0, 2});
    auto value = infinicore::Tensor::empty(
        {num_tokens, num_attention_heads_, v_head_dim_},
        attn_output->dtype(),
        attn_output->device());
    infinicore::op::bmm_strided_(
        value->permute({1, 0, 2}),
        latent_by_head,
        w_uv_);
    auto value_flat = value->view(
        {batch_size, seq_len, num_attention_heads_ * v_head_dim_});

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto output = o_proj_->forward(value_flat);
    if (use_sparse_ && rank_info.tp_size > 1) {
        // Finish sparse attention and the TP output projection before the
        // decoder advances to the next layer.
        infinicore::context::syncStream();
    }
    return output;
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
    infinicore::Tensor compressed;
    infinicore::Tensor q_lora_norm;
    if (q_lora_rank_ == 0) {
        q_linear = q_proj_->forward(hidden_states_mutable);
        compressed = kv_a_proj_with_mqa_->forward(hidden_states_mutable);
    } else {
        auto qkv_a = fused_qkv_a_proj_->forward_split(hidden_states_mutable);
        if (qkv_a.size() != 2) {
            throw std::runtime_error(
                "DeepseekV2MLAAttention fused_qkv_a_proj must return two output slices");
        }
        auto q_a = qkv_a[0];
        compressed = qkv_a[1];
        q_lora_norm = q_a_layernorm_->forward(q_a);
        q_linear = q_b_proj_->forward(q_lora_norm);
    }
    auto q = q_linear->view({seq_len, num_attention_heads_, q_head_dim_});
    auto q_nope = q->narrow({{2, 0, qk_nope_head_dim_}});
    auto q_pe = q->narrow({{2, qk_nope_head_dim_, qk_rope_head_dim_}})->contiguous();

    compressed = compressed->view({seq_len, kv_lora_rank_ + qk_rope_head_dim_});
    auto compressed_kv = compressed->narrow({{1, 0, kv_lora_rank_}})->contiguous();
    auto k_pe = compressed->narrow({{1, kv_lora_rank_, qk_rope_head_dim_}})->contiguous();

    auto pos_ids = position_ids_for_rope_(position_ids);
    auto k_pe_rope = k_pe->view({seq_len, 1, qk_rope_head_dim_});
    infinicore::op::fused_rotary_embedding_(
        q_pe,
        k_pe_rope,
        pos_ids,
        static_cast<int64_t>(qk_rope_head_dim_),
        rope_cos_sin_cache_,
        rotary_emb_->algo() == infinicore::nn::RoPE::Algo::GPT_NEOX);
    debug_dump_dsa(q_nope, "q_nope", layer_idx_);
    debug_dump_dsa(q_pe, "q_pe", layer_idx_);
    debug_dump_dsa(k_pe_rope, "k_pe_rope", layer_idx_);

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto request_ids = attn_metadata.request_ids;
    auto cu_seqlens = attn_metadata.cu_seqlens;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());
    ASSERT(total_sequence_lengths.has_value());
    ASSERT(request_ids.has_value());
    ASSERT(attn_metadata.max_context_len.has_value());

    if (hidden_states->device().getType() != infinicore::Device::Type::ILUVATAR) {
        throw std::runtime_error("DeepseekV2MLAAttention: the vLLM-style MLA cache path currently requires Iluvatar");
    }
    const bool use_fp8_ds_mla = kv_cache->dtype() == infinicore::DataType::U8;
    const bool use_fused_fp8_cache = use_sparse_ && use_fp8_ds_mla
                                  && std::getenv("INFINILM_GLM_DISABLE_FUSED_FP8_MLA_CACHE") == nullptr;
    const bool use_vendor_sparse = use_sparse_ && use_fp8_ds_mla
                                && !forward_context.mla_vendor_cache_vec.empty();
    if (use_vendor_sparse
        && forward_context.mla_vendor_cache_vec.size()
               != forward_context.kv_cache_vec.size()) {
        throw std::runtime_error(
            "GLM FP8 Sparse MLA vendor shadow layer count mismatch");
    }
    if (use_vendor_sparse && !use_fused_fp8_cache) {
        throw std::runtime_error(
            "GLM FP8 Sparse MLA vendor shadow requires fused cache producer");
    }
    infinicore::Tensor kv_norm;
    if (use_fused_fp8_cache) {
        auto rope_cache = k_pe_rope->view({seq_len, qk_rope_head_dim_});
        if (use_vendor_sparse) {
            infinicore::op::fp8_mla_rmsnorm_dual_cache_(
                kv_cache,
                forward_context.mla_vendor_cache_vec[layer_idx_],
                compressed_kv,
                kv_a_layernorm_->weight(),
                rope_cache,
                slot_mapping.value(),
                kv_a_layernorm_->eps());
        } else {
            infinicore::op::fp8_mla_rmsnorm_cache_(
                kv_cache,
                compressed_kv,
                kv_a_layernorm_->weight(),
                rope_cache,
                slot_mapping.value(),
                kv_a_layernorm_->eps());
        }
    } else {
        kv_norm = kv_a_layernorm_->forward(compressed_kv);
        debug_dump_dsa(kv_norm, "kv_norm", layer_idx_);
        infinicore::op::concat_and_cache_mla_(
            kv_norm,
            k_pe_rope->view({seq_len, qk_rope_head_dim_}),
            kv_cache,
            slot_mapping.value(),
            use_fp8_ds_mla ? "fp8_ds_mla" : "auto",
            kv_cache_k_scale_);
    }
    debug_dump_dsa(kv_cache, "mla_cache", layer_idx_);

    const bool is_prefill = seq_len != total_sequence_lengths.value()->shape()[0];
    if (use_sparse_) {
        auto &topk_indices_opt = forward_context.dsa_topk_indices;
        if (!topk_indices_opt.has_value()) {
            throw std::runtime_error("Sparse MLA top-k buffer is not allocated");
        }
        auto topk_indices = topk_indices_opt.value();
        if (!skip_topk_) {
            indexer_->forward(hidden_states, q_lora_norm, pos_ids, topk_indices);
            debug_dump_dsa(topk_indices, "topk_local", layer_idx_);
        }

        const size_t num_tokens = seq_len;
        if (request_ids.value()->numel() < num_tokens) {
            throw std::runtime_error(
                "Sparse MLA request_ids is shorter than the flattened token batch");
        }
        auto token_request_ids = request_ids.value();
        if (token_request_ids->numel() != num_tokens) {
            token_request_ids = token_request_ids->narrow({{0, 0, num_tokens}});
        }
        auto global_indices = infinicore::Tensor::empty(
            topk_indices->shape(), infinicore::DataType::I32, hidden_states->device());
        const int64_t block_size = static_cast<int64_t>(kv_cache->size(1));
        // Match the Iluvatar vLLM production path: without prefill
        // workspace, request-local indices use the decode mapping kernel.
        infinicore::op::map_decode_request_block_indices_(
            global_indices,
            token_request_ids,
            block_tables.value(),
            topk_indices,
            block_size);

        auto topk_lens = infinicore::Tensor::empty(
            {num_tokens}, infinicore::DataType::I32, hidden_states->device());
        auto sparse_indices = global_indices->view(
            {num_tokens, 1, global_indices->size(1)});
        infinicore::op::topk_indices_context_lens_(topk_lens, sparse_indices);
        debug_dump_dsa(global_indices, "topk_global", layer_idx_);
        debug_dump_dsa(topk_lens, "topk_lens", layer_idx_);

        auto q_latent = project_q_nope_to_latent_(q_nope);
        auto query_states = infinicore::op::concat_mla_q(q_latent, q_pe);
        debug_dump_dsa(q_latent, "q_latent", layer_idx_);
        debug_dump_dsa(query_states, "query_states", layer_idx_);
        auto attn_output = infinicore::Tensor::empty(
            {num_tokens, num_attention_heads_, kv_lora_rank_},
            query_states->dtype(),
            query_states->device());
        auto attention_kv_cache = use_vendor_sparse
                                    ? forward_context.mla_vendor_cache_vec[layer_idx_]
                                    : kv_cache;
        auto sparse_kv_cache = attention_kv_cache->view(
            {static_cast<size_t>(
                 attention_kv_cache->size(0) * attention_kv_cache->size(1)),
             1,
             static_cast<size_t>(attention_kv_cache->size(2))});
        infinicore::op::sparse_flash_mla_(
            attn_output,
            query_states,
            sparse_kv_cache,
            sparse_indices,
            topk_lens,
            softmax_scale_);
        debug_dump_dsa(attn_output, "attn_output", layer_idx_);
        auto sparse_output = project_latent_to_value_(attn_output, batch_size, seq_len);
        debug_dump_dsa(sparse_output, "projected", layer_idx_);
        return sparse_output;
    }
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
