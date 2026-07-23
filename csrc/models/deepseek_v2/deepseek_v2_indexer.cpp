#include "deepseek_v2_indexer.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../layers/rotary_embedding/rotary_embedding_factory.hpp"
#include "../../utils.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/dsa.hpp"
#include "infinicore/ops/fp8_indexer_logits.hpp"
#include "infinicore/ops/fp8_indexer_quant.hpp"
#include "infinicore/ops/mul.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace infinilm::models::deepseek_v2 {
namespace {
void debug_dump_indexer(const infinicore::Tensor &tensor,
                        const std::string &name,
                        size_t layer_idx) {
    if (layer_idx != 0 || std::getenv("INFINILM_GLM_DEBUG_DUMP") == nullptr) {
        return;
    }
    const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tensor->debug("/tmp/glmdbg_indexer_" + name + "_rank_"
                  + std::to_string(rank.tp_rank) + ".bin");
}
} // namespace

DeepseekV32Indexer::DeepseekV32Indexer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    num_heads_ = model_config->get<size_t>("index_n_heads");
    head_dim_ = model_config->get<size_t>("index_head_dim");
    rope_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    topk_tokens_ = model_config->get<size_t>("index_topk");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = rank_info.tp_rank;
    tp_size_ = rank_info.tp_size;
    communicator_ = rank_info.comm;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t q_lora_rank = model_config->get<size_t>("q_lora_rank");
    const auto &dtype = model_config->get_dtype();
    auto quantization = model_config->get_quantization_method();

    INFINICORE_NN_MODULE_INIT(
        wq_b, q_lora_rank, num_heads_ * head_dim_, quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        wk, hidden_size, head_dim_, quantization, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        weights_proj, hidden_size, num_heads_, false, dtype, device);
    fused_wk_weights_proj_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
        hidden_size, head_dim_ + num_heads_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, 1e-6, dtype, device);

    auto scaling = infinilm::layers::rotary_embedding::make_scaling_config(model_config);
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(
        head_dim_,
        rope_dim_,
        model_config->get<size_t>("max_position_embeddings"),
        model_config->get<double>("rope_theta"),
        infinicore::nn::RoPE::Algo::GPT_J,
        dtype,
        device,
        std::move(scaling));
    cos_sin_cache_ = infinilm::layers::rotary_embedding::get_rope_cos_sin_cache(rotary_emb_);
    weights_scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_ * num_heads_));
    one_i32_ = infinicore::Tensor::ones({1}, infinicore::DataType::I32, device);
}

void DeepseekV32Indexer::process_weights_after_loading() {
    const auto wk_weight = wk_->weight();
    const auto wk_scale = wk_->weight_scale();
    const auto weights_weight = weights_proj_->weight();
    if (!wk_weight || wk_weight->dtype() != infinicore::DataType::I8
        || wk_weight->ndim() != 2) {
        throw std::runtime_error(
            "DeepseekV32Indexer expects wk int8 weight runtime [hidden,head_dim]");
    }
    if (!wk_scale || wk_scale->dtype() != infinicore::DataType::F32
        || wk_scale->ndim() != 2
        || wk_scale->size(0) != wk_weight->size(1)
        || wk_scale->size(1) != 1) {
        throw std::runtime_error(
            "DeepseekV32Indexer expects wk float32 weight_scale [head_dim,1]");
    }
    if (!weights_weight || weights_weight->dtype() != k_norm_->dtype()
        || weights_weight->ndim() != 2
        || weights_weight->size(1) != wk_weight->size(0)) {
        throw std::runtime_error(
            "DeepseekV32Indexer expects BF16 weights_proj [num_heads,hidden]");
    }

    auto wk_out_in = wk_weight->permute({1, 0})->contiguous();
    auto wk_f32 = infinicore::Tensor::empty(
        wk_out_in->shape(), infinicore::DataType::F32, wk_out_in->device());
    infinicore::op::cast_(wk_f32, wk_out_in);
    auto expanded_scale = infinicore::op::broadcast_to(
        wk_scale,
        {static_cast<int64_t>(wk_out_in->size(0)),
         static_cast<int64_t>(wk_out_in->size(1))});
    auto dequantized_f32 = infinicore::op::mul(wk_f32, expanded_scale);
    auto dequantized = infinicore::Tensor::empty(
        wk_out_in->shape(), weights_weight->dtype(), wk_out_in->device());
    infinicore::op::cast_(dequantized, dequantized_f32);

    auto fused_weight = infinicore::op::cat({dequantized, weights_weight}, 0);
    fused_wk_weights_proj_->weight()->copy_from(fused_weight);
    wk_->release_parameters();
    weights_proj_->release_parameters();
}

void DeepseekV32Indexer::sync_topk_indices_(infinicore::Tensor topk_indices) const {
    if (tp_size_ == 1) {
        return;
    }
    if (communicator_ == nullptr) {
        throw std::runtime_error("DeepseekV32Indexer requires a TP communicator");
    }

    if (tp_rank_ != 0) {
        set_zeros_device_async(topk_indices);
    }
    infinicore::op::distributed::allreduce_(
        topk_indices, topk_indices, INFINICCL_SUM, communicator_);
}

void DeepseekV32Indexer::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &q_lora,
    const infinicore::Tensor &positions,
    infinicore::Tensor topk_indices) const {
    const size_t num_tokens = positions->numel();
    if (topk_indices->shape()
        != std::vector<size_t>{num_tokens, topk_tokens_}) {
        throw std::runtime_error("DeepseekV32Indexer: invalid topk buffer shape");
    }

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &metadata = forward_context.attn_metadata;
    if (layer_idx_ >= forward_context.indexer_cache_vec.size()) {
        throw std::runtime_error("DeepseekV32Indexer: indexer cache is not allocated");
    }
    if (!metadata.slot_mapping.has_value()
        || !metadata.input_offsets.has_value()
        || !metadata.cu_seqlens.has_value()
        || !metadata.total_sequence_lengths.has_value()
        || !metadata.request_ids.has_value()
        || !metadata.block_tables.has_value()
        || !metadata.max_context_len.has_value()) {
        throw std::runtime_error("DeepseekV32Indexer: incomplete paged attention metadata");
    }
    const size_t num_requests = metadata.total_sequence_lengths.value()->numel();

    // Rank-local indexer compute wins once larger batches amortize its fixed
    // GEMM cost. Batch two with long contexts is faster on rank 0 plus a
    // small top-k collective than repeating the indexer on all four TP ranks.
    const bool use_replicated_indexer = num_requests > 2;
    if (!use_replicated_indexer && tp_rank_ != 0) {
        sync_topk_indices_(topk_indices);
        debug_dump_indexer(topk_indices, "selected", layer_idx_);
        return;
    }

    auto hidden_mutable = hidden_states;
    auto q_lora_mutable = q_lora;
    auto q_raw = wq_b_->forward(q_lora_mutable)
                     ->view({num_tokens, num_heads_, head_dim_});
    auto k_weights = fused_wk_weights_proj_->forward(hidden_mutable)
                         ->view({num_tokens, head_dim_ + num_heads_});
    auto k_raw = k_weights->narrow({{1, 0, head_dim_}});
    auto weights_raw = k_weights->narrow({{1, head_dim_, num_heads_}});
    debug_dump_indexer(q_raw, "q_raw", layer_idx_);
    debug_dump_indexer(k_raw, "k_raw", layer_idx_);
    debug_dump_indexer(weights_raw, "weights_raw", layer_idx_);
    debug_dump_indexer(k_weights, "k_weights", layer_idx_);

    auto &k_cache = forward_context.indexer_cache_vec[layer_idx_];
    auto q_fp8 = infinicore::Tensor::empty(
        q_raw->shape(), infinicore::DataType::F8, q_raw->device());
    auto weights_fp32 = infinicore::Tensor::empty(
        {num_tokens, num_heads_}, infinicore::DataType::F32, q_raw->device());
    const bool use_fused_fp8 = std::getenv("INFINILM_GLM_DISABLE_FUSED_FP8_INDEXER") == nullptr;
    if (use_fused_fp8) {
        infinicore::op::fused_fp8_indexer_(
            q_fp8, weights_fp32, k_cache, q_raw, k_weights,
            k_norm_->weight(), k_norm_->bias(), positions, cos_sin_cache_,
            metadata.slot_mapping.value(), rope_dim_, k_norm_->eps(),
            weights_scale_);
    } else {
        auto q = infinicore::Tensor::empty(
            q_raw->shape(), q_raw->dtype(), q_raw->device());
        auto k = infinicore::Tensor::empty(
            {num_tokens, head_dim_}, q_raw->dtype(), q_raw->device());
        auto weights = infinicore::Tensor::empty(
            {num_tokens, num_heads_}, q_raw->dtype(), q_raw->device());
        auto empty_cache = infinicore::Tensor::empty(
            {0}, q_raw->dtype(), q_raw->device());
        auto empty_slots = infinicore::Tensor::empty(
            {0}, infinicore::DataType::I64, q_raw->device());
        infinicore::op::fused_deepseek_v2_indexer_postprocess_(
            q, k, weights, empty_cache, empty_slots, q_raw, k_weights,
            k_norm_->weight(), k_norm_->bias(), positions, cos_sin_cache_,
            0, false, k_norm_->eps(), weights_scale_);
        debug_dump_indexer(q, "q", layer_idx_);
        debug_dump_indexer(k, "k", layer_idx_);
        infinicore::op::indexer_k_quant_and_cache_(
            k, k_cache, metadata.slot_mapping.value(),
            static_cast<int64_t>(head_dim_), "ue8m0");
        debug_dump_indexer(weights, "weights", layer_idx_);
        infinicore::op::fp8_indexer_quant_(
            q_fp8, weights_fp32, q, weights);
    }
    debug_dump_indexer(k_cache, "k_cache", layer_idx_);
    debug_dump_indexer(q_fp8, "q_fp8", layer_idx_);
    debug_dump_indexer(weights_fp32, "weights_fp32", layer_idx_);

    const int64_t max_context_len = metadata.max_context_len.value();
    auto logits = infinicore::Tensor::empty(
        {num_tokens, static_cast<size_t>(max_context_len)},
        infinicore::DataType::F32,
        q_raw->device());
    const bool is_prefill = num_tokens != num_requests;
    infinicore::op::fp8_indexer_logits_(
        logits, q_fp8, k_cache, metadata.block_tables.value(), weights_fp32,
        positions, metadata.request_ids.value());
    debug_dump_indexer(logits, "logits", layer_idx_);

    if (is_prefill) {
        auto cu_seqlen_ks = infinicore::Tensor::empty(
            {num_tokens}, infinicore::DataType::I32, q_raw->device());
        set_zeros_device_async(cu_seqlen_ks);
        auto positions_i32 = infinicore::Tensor::empty(
            {num_tokens}, infinicore::DataType::I32, q_raw->device());
        infinicore::op::cast_(positions_i32, positions);
        auto ones = infinicore::op::broadcast_to(
                        one_i32_, {static_cast<int64_t>(num_tokens)})
                        ->contiguous();
        auto cu_seqlen_ke = infinicore::op::add(positions_i32, ones);
        infinicore::op::select_prefill_topk_block_indices_(
            topk_indices, logits, cu_seqlen_ks, cu_seqlen_ke);
    } else {
        infinicore::op::select_decode_topk_block_indices_(
            topk_indices, logits, metadata.total_sequence_lengths.value());
    }
    if (!use_replicated_indexer) {
        sync_topk_indices_(topk_indices);
    }
    debug_dump_indexer(topk_indices, "selected", layer_idx_);
}

} // namespace infinilm::models::deepseek_v2
