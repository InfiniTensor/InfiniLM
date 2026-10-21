#pragma once

#include "../../config/model_config.hpp"
#include "../../global_state/piecewise_prefill_state.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "minicpm5_moe_attention.hpp"
#include "minicpm5_moe_sparse_moe_block.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <cstddef>
#include <memory>
#include <tuple>

namespace infinilm::models::minicpm5_moe {

class MiniCPM5MoEDecoderLayer : public infinicore::nn::Module {
public:
    MiniCPM5MoEDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &positions,
            infinicore::Tensor &hidden_states,
            infinicore::Tensor &residual) const;

    void piecewise_pre_attn(const infinicore::Tensor &positions,
                            infinicore::Tensor &hidden_states,
                            infinicore::Tensor &residual,
                            global_state::PiecewiseLayerStaging &staging) const;

    /// Layernorm + past-independent QKV (CG-capturable). No RoPE.
    void piecewise_pre_attn_qkv(const infinicore::Tensor &positions,
                                infinicore::Tensor &hidden_states,
                                infinicore::Tensor &residual,
                                global_state::PiecewiseLayerStaging &staging) const;

    /// Eager RoPE on staging Q/K (host-break).
    void piecewise_pre_attn_rope(const infinicore::Tensor &positions,
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
    INFINICORE_NN_MODULE(MiniCPM5MoEAttention, self_attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, dense_mlp);
    INFINICORE_NN_MODULE(MiniCPM5MoESparseMoeBlock, moe_mlp);
    bool use_moe_{false};
    size_t layer_idx_{0};

private:
    infinicore::Tensor mlp_forward(const infinicore::Tensor &hidden_states) const;
};

} // namespace infinilm::models::minicpm5_moe
