#include "qwen3_5_decoderLayer.hpp"
#include "../../global_state/ar_profile.hpp"
#include "../../global_state/global_state.hpp"
#include "../../global_state/piecewise_inductor_flags.hpp"
#include "../../global_state/piecewise_prefill_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {

Qwen35DecoderLayer::Qwen35DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       size_t layer_idx,
                                       const infinicore::Device &device)
    : layer_idx_(layer_idx) {

    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);

    const std::vector<std::string> layer_types = model_config->get<std::vector<std::string>>("layer_types");
    layer_type_ = layer_types[layer_idx];
    if ("linear_attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(linear_attn, model_config, layer_idx, device);
    } else if ("full_attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    } else {
        throw std::runtime_error("infinilm::models::qwen3_5::Qwen35DecoderLayer: unsupported layer_type '" + layer_type_ + "' for layer " + std::to_string(layer_idx));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Qwen35DecoderLayer::forward(const infinicore::Tensor &positions,
                                                                               infinicore::Tensor &hidden_states,
                                                                               infinicore::Tensor &residual) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    if ("linear_attention" == layer_type_) {
        hidden_states = linear_attn_->forward(hidden_states);
    } else if ("full_attention" == layer_type_) {
        hidden_states = self_attn_->forward(positions, hidden_states);
    }

    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = mlp_->forward(hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor Qwen35DecoderLayer::forward(const infinicore::Tensor &positions,
                                               infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    if ("linear_attention" == layer_type_) {
        hidden_states = linear_attn_->forward(hidden_states);
    } else if ("full_attention" == layer_type_) {
        hidden_states = self_attn_->forward(positions, hidden_states);
    }
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

void Qwen35DecoderLayer::piecewise_pre_attn(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    (void)global_state::piecewise_inductor_segment_enabled();
    piecewise_pre_attn_qkv(positions, hidden_states, residual, staging);
    piecewise_pre_attn_rope(positions, staging);
}

void Qwen35DecoderLayer::piecewise_pre_attn_qkv(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    input_layernorm_->forward_inplace(hidden_states, residual);
    if ("linear_attention" == layer_type_) {
        // GDN: LN only in CG-capturable pre; eager_attn runs the delta-net.
        (void)positions;
        (void)staging;
        return;
    }
    self_attn_->forward_pre_attn_qkv_piecewise(positions, hidden_states, staging);
}

void Qwen35DecoderLayer::piecewise_pre_attn_rope(
    const infinicore::Tensor &positions,
    global_state::PiecewiseLayerStaging &staging) const {
    if ("linear_attention" == layer_type_) {
        (void)positions;
        (void)staging;
        return;
    }
    self_attn_->forward_pre_attn_rope_piecewise(positions, staging);
}

void Qwen35DecoderLayer::piecewise_eager_attn(
    const infinicore::Tensor &positions,
    global_state::PiecewiseLayerStaging &staging) const {
    if ("linear_attention" == layer_type_) {
        // Prefer working eager-GDN over CG-capturing GDN (SSM/conv state).
        auto &piecewise = global_state::get_forward_context().piecewise;
        auto &hidden = piecewise.hidden_states;
        const size_t seq_len = hidden->size(1);
        const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;
        auto hs = hidden->narrow({{1, 0, valid_len}});
        auto out = linear_attn_->forward(hs);
        hidden->narrow({{1, 0, valid_len}})->copy_from(out);
        if (valid_len < seq_len) {
            auto tail = hidden->narrow({{1, valid_len, seq_len - valid_len}});
            set_zeros(tail);
        }
        (void)positions;
        (void)staging;
        return;
    }
    self_attn_->forward_eager_attn_piecewise(positions, staging);
}

void Qwen35DecoderLayer::piecewise_post_attn_cg(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t seq_len = hidden_states->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : seq_len;
    if ("linear_attention" == layer_type_) {
        // Skip o_proj — GDN already wrote into hidden_states in eager_attn.
        (void)staging;
        post_attention_layernorm_->forward_inplace(hidden_states, residual);
        auto hidden_for_mlp = (valid_len < seq_len)
                                  ? hidden_states->narrow({{1, 0, valid_len}})
                                  : hidden_states;
        auto mlp_out = mlp_->forward_matmul(hidden_for_mlp);
        mlp_->forward_allreduce(mlp_out);
        if (valid_len < seq_len) {
            hidden_states->narrow({{1, 0, valid_len}})->copy_from(mlp_out);
        } else {
            hidden_states->copy_from(mlp_out);
        }
        return;
    }
    self_attn_->forward_post_attn_piecewise_cg_into(hidden_states, staging);
    self_attn_->forward_post_attn_piecewise_allreduce_into(hidden_states, staging);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    auto hidden_for_mlp = (valid_len < seq_len)
                              ? hidden_states->narrow({{1, 0, valid_len}})
                              : hidden_states;
    auto mlp_out = mlp_->forward_matmul(hidden_for_mlp);
    mlp_->forward_allreduce(mlp_out);
    if (valid_len < seq_len) {
        hidden_states->narrow({{1, 0, valid_len}})->copy_from(mlp_out);
    } else {
        hidden_states->copy_from(mlp_out);
    }
}

void Qwen35DecoderLayer::piecewise_post_attn_allreduce(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    (void)residual;
    (void)hidden_states;
    (void)staging;
    // AllReduce already applied inside piecewise_post_attn_cg.
}

} // namespace infinilm::models::qwen3_5
