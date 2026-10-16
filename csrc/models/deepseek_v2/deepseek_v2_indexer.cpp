#include "deepseek_v2_indexer.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/quantization/none_quantization.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../layers/rotary_embedding/rotary_embedding_factory.hpp"
#include "../../utils.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/distributed/broadcast.hpp"
#include "infinicore/ops/dsa.hpp"
#include "infinicore/ops/fp8_indexer_logits.hpp"
#include "infinicore/ops/fp8_indexer_quant.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace infinilm::models::deepseek_v2 {

DeepseekV32Indexer::DeepseekV32Indexer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx, const infinicore::Device &device) {
  layer_idx_ = layer_idx;
  num_heads_ = model_config->get<size_t>("index_n_heads");
  head_dim_ = model_config->get<size_t>("index_head_dim");
  rope_dim_ = model_config->get<size_t>("qk_rope_head_dim");
  topk_tokens_ = model_config->get<size_t>("index_topk");
  const auto &rank_info =
      infinilm::global_state::get_tensor_model_parallel_rank_info();
  tp_rank_ = rank_info.tp_rank;
  tp_size_ = rank_info.tp_size;
  communicator_ = rank_info.comm;
  const size_t hidden_size = model_config->get<size_t>("hidden_size");
  const size_t q_lora_rank = model_config->get<size_t>("q_lora_rank");
  const auto &dtype = model_config->get_dtype();
  auto quantization =
      std::make_shared<infinilm::quantization::NoneQuantization>();

  INFINICORE_NN_MODULE_INIT(wq_b, q_lora_rank, num_heads_ * head_dim_,
                            quantization, false, dtype, device);
  INFINICORE_NN_MODULE_INIT(wk, hidden_size, head_dim_, quantization, false,
                            dtype, device);
  INFINICORE_NN_MODULE_INIT(weights_proj, hidden_size, num_heads_, false, dtype,
                            device);
  INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, 1e-6, dtype, device);

  auto scaling =
      infinilm::layers::rotary_embedding::make_scaling_config(model_config);
  rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(
      head_dim_, rope_dim_,
      model_config->get<size_t>("max_position_embeddings"),
      model_config->get<double>("rope_theta"),
      infinicore::nn::RoPE::Algo::GPT_J, dtype, device, std::move(scaling));
  cos_sin_cache_ =
      infinilm::layers::rotary_embedding::get_rope_cos_sin_cache(rotary_emb_);
  weights_scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_ * num_heads_));
  one_i32_ = infinicore::Tensor::ones({1}, infinicore::DataType::I32, device);
}

void DeepseekV32Indexer::process_weights_after_loading() {}

void DeepseekV32Indexer::sync_topk_indices_(
    infinicore::Tensor topk_indices) const {
  if (tp_size_ == 1) {
    return;
  }
  if (communicator_ == nullptr) {
    throw std::runtime_error("DeepseekV32Indexer requires a TP communicator");
  }

  infinicore::op::distributed::broadcast_(topk_indices, topk_indices, 0,
                                          communicator_);
}

void DeepseekV32Indexer::forward(const infinicore::Tensor &hidden_states,
                                 const infinicore::Tensor &q_lora,
                                 const infinicore::Tensor &positions,
                                 infinicore::Tensor topk_indices) const {
  const size_t num_tokens = positions->numel();
  if (topk_indices->shape() != std::vector<size_t>{num_tokens, topk_tokens_}) {
    throw std::runtime_error("DeepseekV32Indexer: invalid topk buffer shape");
  }

  auto &forward_context = infinilm::global_state::get_forward_context();
  auto &metadata = forward_context.attn_metadata;
  if (layer_idx_ >= forward_context.indexer_cache_vec.size()) {
    throw std::runtime_error(
        "DeepseekV32Indexer: indexer cache is not allocated");
  }
  if (!metadata.slot_mapping.has_value() ||
      !metadata.input_offsets.has_value() || !metadata.cu_seqlens.has_value() ||
      !metadata.total_sequence_lengths.has_value() ||
      !metadata.request_ids.has_value() || !metadata.block_tables.has_value() ||
      !metadata.max_context_len.has_value()) {
    throw std::runtime_error(
        "DeepseekV32Indexer: incomplete paged attention metadata");
  }
  const size_t num_requests = metadata.total_sequence_lengths.value()->numel();

  // Rank-local indexer compute wins once larger batches amortize its fixed
  // GEMM cost. Batch two with long contexts is faster on rank 0 plus a
  // small top-k collective than repeating the indexer on all four TP ranks.
  const bool use_replicated_indexer = num_requests > 2;
  if (!use_replicated_indexer && tp_rank_ != 0) {
    sync_topk_indices_(topk_indices);
    return;
  }

  auto hidden_mutable = hidden_states;
  auto q_lora_mutable = q_lora;
  auto q_raw =
      wq_b_->forward(q_lora_mutable)->view({num_tokens, num_heads_, head_dim_});
  auto k_raw = wk_->forward(hidden_mutable)->view({num_tokens, head_dim_});
  auto weights_raw =
      weights_proj_->forward(hidden_mutable)->view({num_tokens, num_heads_});
  auto k_weights = infinicore::op::cat({k_raw, weights_raw}, 1);

  auto &k_cache = forward_context.indexer_cache_vec[layer_idx_];
  auto q_fp8 = infinicore::Tensor::empty(
      q_raw->shape(), infinicore::DataType::F8, q_raw->device());
  auto weights_fp32 = infinicore::Tensor::empty(
      {num_tokens, num_heads_}, infinicore::DataType::F32, q_raw->device());
  const bool use_fused_fp8 =
      std::getenv("INFINILM_GLM_DISABLE_FUSED_FP8_INDEXER") == nullptr;
  if (use_fused_fp8) {
    infinicore::op::fused_fp8_indexer_(
        q_fp8, weights_fp32, k_cache, q_raw, k_weights, k_norm_->weight(),
        k_norm_->bias(), positions, cos_sin_cache_,
        metadata.slot_mapping.value(), rope_dim_, k_norm_->eps(),
        weights_scale_);
  } else {
    auto q = infinicore::Tensor::empty(q_raw->shape(), q_raw->dtype(),
                                       q_raw->device());
    auto k = infinicore::Tensor::empty({num_tokens, head_dim_}, q_raw->dtype(),
                                       q_raw->device());
    auto weights = infinicore::Tensor::empty({num_tokens, num_heads_},
                                             q_raw->dtype(), q_raw->device());
    auto empty_cache =
        infinicore::Tensor::empty({0}, q_raw->dtype(), q_raw->device());
    auto empty_slots = infinicore::Tensor::empty({0}, infinicore::DataType::I64,
                                                 q_raw->device());
    infinicore::op::fused_deepseek_v2_indexer_postprocess_(
        q, k, weights, empty_cache, empty_slots, q_raw, k_weights,
        k_norm_->weight(), k_norm_->bias(), positions, cos_sin_cache_, 0, false,
        k_norm_->eps(), weights_scale_);
    infinicore::op::indexer_k_quant_and_cache_(
        k, k_cache, metadata.slot_mapping.value(),
        static_cast<int64_t>(head_dim_), "ue8m0");
    infinicore::op::fp8_indexer_quant_(q_fp8, weights_fp32, q, weights);
  }

  const int64_t max_context_len = metadata.max_context_len.value();
  auto logits = infinicore::Tensor::empty(
      {num_tokens, static_cast<size_t>(max_context_len)},
      infinicore::DataType::F32, q_raw->device());
  const bool is_prefill = num_tokens != num_requests;
  infinicore::op::fp8_indexer_logits_(
      logits, q_fp8, k_cache, metadata.block_tables.value(), weights_fp32,
      positions, metadata.request_ids.value());

  if (is_prefill) {
    auto cu_seqlen_ks = infinicore::Tensor::empty(
        {num_tokens}, infinicore::DataType::I32, q_raw->device());
    set_zeros_device_async(cu_seqlen_ks);
    auto positions_i32 = infinicore::Tensor::empty(
        {num_tokens}, infinicore::DataType::I32, q_raw->device());
    infinicore::op::cast_(positions_i32, positions);
    auto ones = infinicore::op::broadcast_to(one_i32_,
                                             {static_cast<int64_t>(num_tokens)})
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
}

} // namespace infinilm::models::deepseek_v2
