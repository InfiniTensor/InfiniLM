#include "clover_gdn.hpp"
#include "../../global_state/forward_context.hpp"
#include "execution/submit.hpp"
#include "runtime/runtime.hpp"
#include <base/clover_block_conv1d.h>
#include <base/clover_local_memory.h>
#include <base/clover_recurrent_gdn.h>

namespace infinilm::models::clover_moe {

CloverGatedDeltaNet::CloverGatedDeltaNet(runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> config,
                                         size_t layer, const engine::distributed::RankInfo &rank,
                                         backends::AttentionBackend backend, const layers::moe::EPConfig &ep)
    : Qwen3NextGatedDeltaNet(runtime, config, layer, rank, backend, ep),
      tp_rank_(rank.tp_rank), tp_size_(rank.tp_size),
      full_key_dim_(local_key_dim_ * tp_size_), full_value_dim_(local_value_dim_ * tp_size_) {
    try {
        const auto dtype = config->get_dtype();
        right_weight_ = weights::Weight(runtime, {2 * full_key_dim_ + full_value_dim_, 1, 3}, dtype);
        bind_weight("conv_right_weight", right_weight_);
        const auto heads = config->get<size_t>("linear_num_value_heads");
        logit_scale_ = weights::Weight(runtime, {heads}, dtype, 0, tp_rank_, tp_size_);
        history_bias_ = weights::Weight(runtime, {heads}, dtype, 0, tp_rank_, tp_size_);
        history_c_ = weights::Weight(runtime, {heads}, dtype, 0, tp_rank_, tp_size_);
        bind_weight("bd_softmax_logit_scale", logit_scale_);
        bind_weight("bd_softmax_history_bias", history_bias_);
        bind_weight("bd_softmax_history_c", history_c_);
    } catch (...) {
        runtime.syncStream();
        throw;
    }
}

void CloverGatedDeltaNet::process_weights_after_loading(runtime::Runtime &runtime, bool) {
    conv_weight_ = conv1d_->state_dict().at("weight").tensor();
    auto local = runtime::Tensor::empty(runtime, {2 * local_key_dim_ + local_value_dim_, 1, 3}, right_weight_->dtype());
    for (size_t part = 0; part < 3; ++part) {
        const size_t width = part == 2 ? local_value_dim_ : local_key_dim_;
        const size_t src = part * full_key_dim_ + tp_rank_ * width;
        const size_t dst = part * local_key_dim_;
        local->narrow({{0, dst, width}})->copy_from(runtime, right_weight_->narrow({{0, src, width}}));
    }
    runtime.syncStream();
    right_weight_ = weights::Weight(local);
    local_weights_["conv_right_weight"] = right_weight_;
}

runtime::Tensor CloverGatedDeltaNet::forward(const runtime::Tensor &hidden,
                                             const global_state::ForwardContext &context) const {
    auto &runtime = context.require_runtime();
    auto input = hidden;
    const size_t tokens = hidden->size(0) * hidden->size(1);
    auto qkv = in_proj_qkv_->forward(runtime, input)->view({1, tokens, 2 * local_key_dim_ + local_value_dim_});
    auto z = in_proj_z_->forward(runtime, input);
    auto a = in_proj_a_->forward(runtime, input)->view({1, tokens, local_num_value_heads_});
    auto b = in_proj_b_->forward(runtime, input)->view({1, tokens, local_num_value_heads_});
    const auto device = runtime.device().type();
    runtime::Tensor mixed;
    if (context.diffusion_preview) {
        auto flat = qkv->view({tokens, qkv->size(2)});
        auto out = runtime::Tensor::empty(runtime, flat->shape(), flat->dtype());
        execution::submit<infini::ops::CloverBlockConv1d>(
            runtime, execution::RecordingPolicy::HostOnly,
            infini::ops::CloverBlockConv1d::DefaultConfig(device),
            flat, context.conv_state_vec[layer_idx_], context.mamba_metadata.init_state_indices.value(),
            conv_weight_, right_weight_.tensor(), context.diffusion_valid, context.diffusion_block_size, out);
        mixed = out->view(qkv->shape());
    } else {
        mixed = conv1d_->forward(qkv, context);
    }
    auto q = mixed->narrow({{2, 0, local_key_dim_}})->as_strided({1, tokens, local_num_key_heads_, key_head_dim_}, {mixed->stride(0), mixed->stride(1), static_cast<runtime::Stride>(key_head_dim_), 1});
    auto k = mixed->narrow({{2, local_key_dim_, local_key_dim_}})->as_strided(q->shape(), q->strides());
    auto v = mixed->narrow({{2, 2 * local_key_dim_, local_value_dim_}})->as_strided({1, tokens, local_num_value_heads_, value_head_dim_}, {mixed->stride(0), mixed->stride(1), static_cast<runtime::Stride>(value_head_dim_), 1});
    auto g = runtime::Tensor::empty(runtime, a->shape(), runtime::DataType::kFloat32);
    auto beta = runtime::Tensor::empty(runtime, a->shape(), runtime::DataType::kFloat32);
    execution::submit<infini::ops::FusedGatedDeltaNetGating>(
        runtime, execution::RecordingPolicy::HostOnly,
        infini::ops::FusedGatedDeltaNetGating::DefaultConfig(device),
        A_log_.tensor(), a, b, dt_bias_.tensor(), 1.0f, 20.0f, g, beta);
    auto out = runtime::Tensor::empty(runtime, v->shape(), v->dtype());
    if (context.diffusion_preview) {
        execution::submit<infini::ops::CloverLocalMemory>(
            runtime, execution::RecordingPolicy::HostOnly,
            infini::ops::CloverLocalMemory::DefaultConfig(device),
            q, k, v, g, context.ssm_state_vec[layer_idx_], context.mamba_metadata.init_state_indices.value(),
            context.diffusion_valid, logit_scale_.tensor(), history_bias_.tensor(), history_c_.tensor(),
            context.diffusion_block_size, out);
    } else {
        execution::submit<infini::ops::CloverRecurrentGdn>(
            runtime, execution::RecordingPolicy::HostOnly,
            infini::ops::CloverRecurrentGdn::DefaultConfig(device),
            q, k, v, g, beta, context.mamba_metadata.input_offsets.value(),
            context.mamba_metadata.init_state_indices.value(), context.mamba_metadata.final_state_indices.value(),
            context.ssm_state_vec[layer_idx_], out);
    }
    auto normalized = norm_->forward(runtime, out->view({tokens * local_num_value_heads_, value_head_dim_}));
    normalized = normalized->view(z->shape());
    auto gate = runtime::Tensor::empty(runtime, z->shape(), z->dtype());
    execution::submit<infini::ops::Silu>(runtime, infini::ops::Silu::DefaultConfig(device), z, gate);
    execution::submit<infini::ops::Mul>(runtime, infini::ops::Mul::DefaultConfig(device), normalized, gate, normalized);
    return out_proj_->forward(runtime, normalized);
}

} // namespace infinilm::models::clover_moe
