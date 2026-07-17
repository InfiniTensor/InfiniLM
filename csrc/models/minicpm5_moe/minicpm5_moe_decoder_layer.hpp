#pragma once

#include "../../global_state/ar_profile.hpp"
#include "../../global_state/piecewise_inductor_flags.hpp"
#include "../../global_state/piecewise_prefill_state.hpp"
#include "minicpm5_moe_attention.hpp"
#include "minicpm5_moe_dense_mlp.hpp"
#include "minicpm5_moe_sparse_moe_block.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops/inductor_segment.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/context/context.hpp"

#include <memory>
#include <stdexcept>
#include <tuple>

namespace infinilm::models::minicpm5_moe {

/**
 * Decoder layer with hybrid dense/MoE MLP and TextDecoderLayer-compatible piecewise_* hooks
 * so PiecewiseTextCausalLM / native CG + inductor work.
 */
class MiniCPM5MoeDecoderLayer : public infinicore::nn::Module {
public:
    MiniCPM5MoeDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

    size_t layer_idx() const { return layer_idx_; }

    infinicore::op::inductor_segment_impl::PreAttnExternalWeightTensors
    pre_attn_external_weights() const {
        auto out = self_attn_->pre_attn_external_weights();
        out.ln_weight = input_layernorm_->weight();
        return out;
    }

    infinicore::op::inductor_segment_impl::MoeExternalWeightTensors
    moe_external_weights() const {
        if (!moe_mlp_) {
            throw std::runtime_error(
                "MiniCPM5MoeDecoderLayer: moe_external_weights on dense MLP layer "
                + std::to_string(layer_idx_));
        }
        return moe_mlp_->moe_external_weights();
    }

    void piecewise_pre_attn(const infinicore::Tensor &positions,
                            infinicore::Tensor &hidden_states,
                            infinicore::Tensor &residual,
                            global_state::PiecewiseLayerStaging &staging) const;

    void piecewise_eager_attn(const infinicore::Tensor &positions,
                              global_state::PiecewiseLayerStaging &staging) const {
        self_attn_->forward_eager_attn_piecewise(positions, staging);
    }

    void piecewise_post_attn(infinicore::Tensor &hidden_states,
                             infinicore::Tensor &residual,
                             global_state::PiecewiseLayerStaging &staging) const {
        piecewise_post_attn_cg(hidden_states, residual, staging);
    }

    void piecewise_post_attn_cg(infinicore::Tensor &hidden_states,
                                infinicore::Tensor &residual,
                                global_state::PiecewiseLayerStaging &staging) const;

    /// Decode-piecewise CG segment: o_proj + post-norm (+ dense MLP). MoE stays eager.
    void piecewise_post_attn_decode_cg(infinicore::Tensor &hidden_states,
                                       infinicore::Tensor &residual,
                                       global_state::PiecewiseLayerStaging &staging) const;

    /// Eager MoE between Graph::run segments (no-op on dense MLP layers).
    void piecewise_eager_moe(infinicore::Tensor &hidden_states,
                             infinicore::Tensor &residual,
                             global_state::PiecewiseLayerStaging &staging) const;

    void piecewise_post_attn_graph(infinicore::Tensor &hidden_states,
                                   infinicore::Tensor &residual,
                                   global_state::PiecewiseLayerStaging &staging) const;

    void piecewise_post_attn_allreduce(infinicore::Tensor &hidden_states,
                                       infinicore::Tensor &residual,
                                       global_state::PiecewiseLayerStaging &staging) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniCPM5MoeAttention, self_attn);

    std::shared_ptr<MiniCPM5DenseMLP> dense_mlp_;
    std::shared_ptr<MiniCPM5MoeSparseMoeBlock> moe_mlp_;

    size_t layer_idx_{0};

private:
    infinicore::Tensor mlp_forward(const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor mlp_forward_matmul_only(const infinicore::Tensor &hidden_states) const;
    void mlp_allreduce_output(infinicore::Tensor &output) const;
};

} // namespace infinilm::models::minicpm5_moe
