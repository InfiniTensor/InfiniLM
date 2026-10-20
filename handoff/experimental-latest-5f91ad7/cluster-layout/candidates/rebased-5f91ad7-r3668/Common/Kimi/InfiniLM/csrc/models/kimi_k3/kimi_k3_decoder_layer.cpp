#include "kimi_k3_decoder_layer.hpp"
#include "../../execution/submit.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"
#include <base/gemm.h>
#include <exception>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace infinilm::models::kimi_k3 {
KimiK3DecoderLayer::KimiK3DecoderLayer(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("KimiK3DecoderLayer construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit KimiK3DecoderLayer construction resources");
    }
    try {
        layer_idx_ = layer_idx;
        qualified_tp_size_ = rank_info.tp_size;
        hidden_size_ = model_config->get<size_t>("hidden_size");
        attn_res_block_size_ = model_config->get<size_t>("attn_res_block_size");
        const auto &dtype = model_config->get_dtype();
        const double eps = model_config->get<double>("rms_norm_eps");
        const auto kda_layers = model_config->get_config_json()
                                    .at("linear_attn_config")
                                    .at("kda_layers")
                                    .get<std::vector<size_t>>();
        is_kda_ = std::find(kda_layers.begin(), kda_layers.end(), layer_idx + 1) != kda_layers.end();
        if (is_kda_) {
            delta_attn_ = std::make_shared<KimiK3DeltaAttention>(
                runtime, model_config, layer_idx, rank_info);
            this->bind_child("self_attn", delta_attn_);
        } else {
            mla_attn_ = std::make_shared<KimiK3MLAAttention>(
                runtime, model_config, layer_idx, rank_info, attention_backend);
            this->bind_child("self_attn", mla_attn_);
        }

        const size_t first_dense_layer = model_config->get<size_t>("first_k_dense_replace");
        const size_t moe_frequency = model_config->get<size_t>("moe_layer_freq");
        use_moe_ = layer_idx >= first_dense_layer && layer_idx % moe_frequency == 0;
        if (use_moe_) {
            block_sparse_moe_ = std::make_shared<std::remove_reference_t<decltype(*block_sparse_moe_)>>(
                runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
            this->bind_child("block_sparse_moe", block_sparse_moe_);
        } else {
            mlp_ = std::make_shared<std::remove_reference_t<decltype(*mlp_)>>(
                runtime, model_config, model_config->get<size_t>("intermediate_size"), rank_info);
            this->bind_child("mlp", mlp_);
        }

        input_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*input_layernorm_)>>(
            runtime, hidden_size_, eps, dtype);
        this->bind_child("input_layernorm", input_layernorm_);
        post_attention_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*post_attention_layernorm_)>>(
            runtime, hidden_size_, eps, dtype);
        this->bind_child("post_attention_layernorm", post_attention_layernorm_);
        self_attention_res_norm_ = std::make_shared<std::remove_reference_t<decltype(*self_attention_res_norm_)>>(
            runtime, hidden_size_, eps, dtype);
        this->bind_child("self_attention_res_norm", self_attention_res_norm_);
        mlp_res_norm_ = std::make_shared<std::remove_reference_t<decltype(*mlp_res_norm_)>>(
            runtime, hidden_size_, eps, dtype);
        this->bind_child("mlp_res_norm", mlp_res_norm_);
        self_attention_res_proj_ = std::make_shared<std::remove_reference_t<decltype(*self_attention_res_proj_)>>(
            runtime, hidden_size_, 1, false, dtype);
        this->bind_child("self_attention_res_proj", self_attention_res_proj_);
        mlp_res_proj_ = std::make_shared<std::remove_reference_t<decltype(*mlp_res_proj_)>>(
            runtime, hidden_size_, 1, false, dtype);
        this->bind_child("mlp_res_proj", mlp_res_proj_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3DecoderLayer::apply_attn_res(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &prefix_sum,
    const infinilm::runtime::Tensor &block_residual_storage,
    size_t block_residual_count,
    const std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> &proj,
    const std::shared_ptr<infinilm::layers::RMSNorm> &norm) const {
    const auto shape = prefix_sum->shape();
    if (block_residual_count >= block_residual_storage->size(1)) {
        throw std::runtime_error("KimiK3DecoderLayer: block residual scratch slot is unavailable");
    }
    auto values = block_residual_storage->narrow(
        {{1, 0, block_residual_count + 1}});
    auto normalized = norm->forward(runtime, values);
    auto scores = proj->forward(runtime, normalized);
    infinilm::execution::submit<infini::ops::Softmax>(
        runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), scores, std::int64_t{1}, std::optional<infini::ops::DataType>{}, scores);
    auto score_matrix = scores->view(
        {scores->size(0), scores->size(2), scores->size(1)});
    auto matmul_output_shape = score_matrix->shape();
    matmul_output_shape[score_matrix->ndim() - 1] = values->size(score_matrix->ndim() - 1);
    auto matmul_output = infinilm::runtime::Tensor::empty(runtime, matmul_output_shape, score_matrix->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), score_matrix, values,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, matmul_output);
    INFINILM_TRACE_OP(Gemm, matmul_output);
    return matmul_output
        ->squeeze(1)
        ->view(shape);
}

infinilm::runtime::Tensor KimiK3DecoderLayer::forward(
    const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &block_residual_storage,
    size_t &block_residual_count,
    global_state::ForwardContext &forward_context) const {
    auto &checked_runtime = forward_context.require_runtime();
    const bool qualified_dense = layer_idx_ == 0 && !use_moe_ && block_residual_count == 0;
    const bool qualified_moe = layer_idx_ == 1 && use_moe_ && block_residual_count == 1;
    const bool qualified_mla = layer_idx_ == 3 && !is_kda_ && use_moe_ && block_residual_count == 1 && qualified_tp_size_ == 8;
    if (checked_runtime.device().type() != infinilm::runtime::Device::Type::kHygon || (!is_kda_ && !qualified_mla) || (qualified_tp_size_ != 1 && qualified_tp_size_ != 8) || !(qualified_dense || qualified_moe || qualified_mla))
        throw std::runtime_error("Kimi decoder qualifies Hygon BF16 TP1/TP8 dense layer0 and KDA-MoE layer1, and TP8 MLA-MoE layer3; other decoder integration remains pending");
    if (!hidden_states || hidden_states->ndim() != 3 || hidden_states->size(0) != 1 || !hidden_states->size(1) || hidden_states->size(2) != hidden_size_ || hidden_states->dtype() != infinilm::runtime::DataType::kBFloat16 || hidden_states->device() != checked_runtime.device() || !block_residual_storage || block_residual_storage->ndim() != 3 || block_residual_storage->size(0) != hidden_states->size(1) || block_residual_storage->size(1) < 2 || block_residual_storage->size(2) != hidden_size_ || block_residual_storage->dtype() != hidden_states->dtype() || block_residual_storage->device() != checked_runtime.device())
        throw std::invalid_argument("Kimi KDA decoder input/residual contract differs");
    auto &runtime = *forward_context.runtime;
    const auto shape = hidden_states->shape();
    auto prefix_sum = hidden_states;
    auto current = hidden_states;

    if (block_residual_count > 0) {
        current = apply_attn_res(runtime, prefix_sum, block_residual_storage,
                                 block_residual_count,
                                 self_attention_res_proj_, self_attention_res_norm_);
    }
    bool reset_prefix = layer_idx_ % attn_res_block_size_ == 0;
    if (reset_prefix) {
        if (block_residual_count + 1 >= block_residual_storage->size(1)) {
            throw std::runtime_error("KimiK3DecoderLayer: block residual storage is full");
        }
        // The current scratch slot becomes a persistent block residual. The
        // attention result will be written into the next scratch slot below.
        ++block_residual_count;
    }

    current = input_layernorm_->forward(runtime, current);
    current = is_kda_ ? delta_attn_->forward(current, forward_context) : mla_attn_->forward(current, forward_context);
    if (reset_prefix) {
        auto scratch = block_residual_storage
                           ->narrow({{1, block_residual_count, 1}})
                           ->view({shape[0], shape[1], hidden_size_});
        scratch->copy_from(runtime, current);
        prefix_sum = scratch;
    } else {
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), prefix_sum, current, prefix_sum);
        INFINILM_TRACE_OP(Add, prefix_sum);
    }

    current = apply_attn_res(runtime, prefix_sum, block_residual_storage,
                             block_residual_count, mlp_res_proj_, mlp_res_norm_);
    current = post_attention_layernorm_->forward(runtime, current);
    current = use_moe_ ? block_sparse_moe_->forward(runtime, current) : mlp_->forward(runtime, current);
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), prefix_sum, current, prefix_sum);
    INFINILM_TRACE_OP(Add, prefix_sum);
    return prefix_sum;
}

} // namespace infinilm::models::kimi_k3
