#pragma once

#include "../../config/model_config.hpp"
#include "../../engine/compiled_prefill_flags.hpp"
#include "../../global_state/ar_profile.hpp"
#include "../../global_state/global_state.hpp"
#include "../../global_state/piecewise_prefill_state.hpp"
#include "../../utils.hpp"
#include "../../utils/layer_hidden_dump.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <tuple>
namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Generic Text decoder layer (transformer block) class.
 *
 * @tparam Attention: The attention module type (e.g., Qwen3Attention)
 * @tparam MLP: The MLP module type (e.g., Qwen3MLP)
 */
template <typename Attention, typename MLP>
class TextDecoderLayer : public infinicore::nn::Module {
public:
    TextDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device)
        : layer_idx_(layer_idx) {

        const auto &dtype{model_config->get_dtype()};
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("input_layernorm", hidden_size, rms_norm_eps, dtype, device);
        post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);

        self_attn_ = this->register_module<Attention>("self_attn", model_config, layer_idx, device);
        mlp_ = this->register_module<MLP>("mlp", model_config, device);
    }

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual) {
        size_t valid_len = 0;
        const auto &piecewise = global_state::get_forward_context().piecewise;
        if (piecewise.valid_seq_len > 0) {
            valid_len = piecewise.valid_seq_len;
        }
        input_layernorm_->forward_inplace(hidden_states, residual);
        infinilm::utils::eager_dump_barrier("eager_dump_post_pre_attn");
        infinilm::utils::dump_layer_hidden(
            hidden_states, layer_idx_, valid_len, "post_pre_attn");
        hidden_states = self_attn_->forward(positions, hidden_states);
        post_attention_layernorm_->forward_inplace(hidden_states, residual);
        hidden_states = mlp_->forward(hidden_states);
        infinilm::utils::eager_dump_barrier("eager_dump_post_mlp");
        infinilm::utils::dump_layer_hidden(
            hidden_states, layer_idx_, valid_len, "post_mlp");
        return std::make_tuple(hidden_states, residual);
    }

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states) {
        auto residual = hidden_states;
        hidden_states = input_layernorm_->forward(hidden_states);
        hidden_states = self_attn_->forward(positions, hidden_states);
        hidden_states = infinicore::op::add(residual, hidden_states);

        residual = hidden_states;
        hidden_states = post_attention_layernorm_->forward(hidden_states);
        hidden_states = mlp_->forward(hidden_states);
        hidden_states = infinicore::op::add(residual, hidden_states);
        return hidden_states;
    }

    size_t layer_idx() const { return layer_idx_; }

    void piecewise_pre_attn_layernorm(infinicore::Tensor &hidden_states,
                                      infinicore::Tensor &residual) const {
        auto &piecewise = global_state::get_forward_context().piecewise;
        const size_t bucket = hidden_states->size(1);
        const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : bucket;
        if (valid_len < bucket) {
            auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
            auto residual_narrow = residual->narrow({{1, 0, valid_len}});
            input_layernorm_->forward_inplace(hidden_narrow, residual_narrow);
        } else {
            input_layernorm_->forward_inplace(hidden_states, residual);
        }
    }

    void piecewise_pre_attn(const infinicore::Tensor &positions,
                            infinicore::Tensor &hidden_states,
                            infinicore::Tensor &residual,
                            global_state::PiecewiseLayerStaging &staging) const {
        piecewise_pre_attn_layernorm(hidden_states, residual);
        self_attn_->forward_pre_attn_piecewise(positions, hidden_states, staging);
    }

    void piecewise_pre_attn_rope(const infinicore::Tensor &positions,
                                 global_state::PiecewiseLayerStaging &staging) const {
        self_attn_->forward_pre_attn_piecewise_apply_rope(positions, staging);
    }

    void piecewise_pre_attn_staging(infinicore::Tensor &hidden_states,
                                    global_state::PiecewiseLayerStaging &staging) const {
        self_attn_->forward_pre_attn_piecewise_fill_staging(hidden_states, staging);
    }

    void piecewise_eager_attn(const infinicore::Tensor &positions,
                              global_state::PiecewiseLayerStaging &staging) const {
        self_attn_->forward_eager_attn_piecewise(positions, staging);
    }

    void piecewise_post_attn(infinicore::Tensor &hidden_states,
                             infinicore::Tensor &residual,
                             global_state::PiecewiseLayerStaging &staging) const {
        piecewise_post_attn_graph(hidden_states, residual, staging);
        piecewise_post_attn_allreduce(hidden_states, residual, staging);
    }

    void piecewise_post_attn_graph(infinicore::Tensor &hidden_states,
                                   infinicore::Tensor &residual,
                                   global_state::PiecewiseLayerStaging &staging) const {
        (void)residual;
        // Graph segment: o_proj matmul (+ in-graph AR when INFINI_PIECEWISE_AR_IN_GRAPH=1).
        self_attn_->forward_post_attn_piecewise_graph_into(hidden_states, staging);
    }

    void piecewise_o_proj_staging(infinicore::Tensor &hidden_states,
                                  global_state::PiecewiseLayerStaging &staging) const {
        self_attn_->forward_post_attn_piecewise_matmul_staging(hidden_states, staging);
    }

    void piecewise_post_attn_mlp_graph(infinicore::Tensor &hidden_states,
                                       infinicore::Tensor &residual) const {
        auto &piecewise = global_state::get_forward_context().piecewise;
        const auto &ctx = global_state::get_forward_context();
        const size_t bucket = hidden_states->size(1);
        const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : bucket;
        if (valid_len < bucket) {
            auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
            auto residual_narrow = residual->narrow({{1, 0, valid_len}});
            post_attention_layernorm_->forward_inplace(hidden_narrow, residual_narrow);
            auto mlp_out = mlp_->forward_matmul_only(hidden_narrow);
            if (ctx.defer_row_parallel_allreduce && mlp_->needs_allreduce()) {
                hidden_narrow->copy_from(mlp_out);
                global_state::ar_profile::allreduce_hidden_valid_contiguous(
                    hidden_states,
                    valid_len,
                    piecewise.ar_staging_mlp,
                    [&](infinicore::Tensor &t) { mlp_->allreduce_output(t); });
            } else if (engine::piecewise_ar_in_graph()
                       && piecewise.phase == global_state::PiecewiseCapturePhase::PostAttnMlp
                       && mlp_->needs_allreduce()) {
                auto staging_narrow = piecewise.ar_staging_mlp->narrow({{1, 0, valid_len}});
                staging_narrow->copy_from(mlp_out);
                mlp_->allreduce_output(staging_narrow);
                hidden_narrow->copy_from(staging_narrow);
            } else {
                hidden_narrow->copy_from(mlp_out);
            }
            infinicore::Tensor hidden_tail = hidden_states->narrow({{1, valid_len, bucket - valid_len}});
            infinicore::Tensor residual_tail = residual->narrow({{1, valid_len, bucket - valid_len}});
            set_zeros(hidden_tail);
            set_zeros(residual_tail);
        } else {
            post_attention_layernorm_->forward_inplace(hidden_states, residual);
            auto mlp_out = mlp_->forward_matmul_only(hidden_states);
            if (ctx.defer_row_parallel_allreduce && mlp_->needs_allreduce()) {
                hidden_states->copy_from(mlp_out);
                global_state::ar_profile::allreduce_hidden_valid_contiguous(
                    hidden_states,
                    valid_len,
                    piecewise.ar_staging_mlp,
                    [&](infinicore::Tensor &t) { mlp_->allreduce_output(t); });
            } else if (engine::piecewise_ar_in_graph()
                       && piecewise.phase == global_state::PiecewiseCapturePhase::PostAttnMlp
                       && mlp_->needs_allreduce()) {
                auto staging_narrow = piecewise.ar_staging_mlp->narrow({{1, 0, valid_len}});
                staging_narrow->copy_from(mlp_out);
                mlp_->allreduce_output(staging_narrow);
                hidden_states->copy_from(staging_narrow);
            } else {
                hidden_states->copy_from(mlp_out);
            }
        }
    }

    void piecewise_post_attn_allreduce(infinicore::Tensor &hidden_states,
                                       infinicore::Tensor &residual,
                                       global_state::PiecewiseLayerStaging &staging) const {
        auto &piecewise = global_state::get_forward_context().piecewise;
        const auto &ctx = global_state::get_forward_context();
        const size_t bucket = hidden_states->size(1);
        const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : bucket;
        const auto step = piecewise.post_attn_replay_step;
        const bool skip_staging_ar = ctx.defer_row_parallel_allreduce;

        if (!skip_staging_ar
            && (step == global_state::PiecewisePostAttnReplayStep::Full
                || step == global_state::PiecewisePostAttnReplayStep::OProjAllreduce)) {
            self_attn_->forward_post_attn_piecewise_allreduce_into(hidden_states, staging);
        }

        if (step == global_state::PiecewisePostAttnReplayStep::Full) {
            piecewise_post_attn_mlp_graph(hidden_states, residual);
        }

        if (!skip_staging_ar
            && !engine::piecewise_ar_in_graph()
            && (step == global_state::PiecewisePostAttnReplayStep::Full
                || step == global_state::PiecewisePostAttnReplayStep::MlpAllreduce)) {
            global_state::ar_profile::allreduce_hidden_valid_contiguous(
                hidden_states,
                valid_len,
                piecewise.ar_staging_mlp,
                [&](infinicore::Tensor &t) { mlp_->allreduce_output(t); });
        }
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(Attention, self_attn);
    INFINICORE_NN_MODULE(MLP, mlp);

    size_t layer_idx_;
};

} // namespace infinilm::layers::causal_lm_templates
