#pragma once

#include "minimax_text_01_attention.hpp"
#include "minimax_text_01_linear_attention.hpp"
#include "minimax_text_01_sparse_moe_block.hpp"

namespace infinilm::models::minimax_text_01 {

/**
 * @brief One decoder (transformer) block of MiniMax-Text-01.
 *
 * Each block contains two layer norms, one attention module and one MoE
 * feed-forward module. The attention type is decided per layer by
 * `attn_type_list[layer_idx]` from the model config (0 = linear attention,
 * 1 = full attention). Full attention uses `MiniMaxText01Attention` and
 * linear (Lightning) attention uses `MiniMaxText01LinearAttention`.
 *
 * The block follows the reference `modeling_minimax_text_01.py`: it combines
 * the residual path with the sublayer output using learnable alpha/beta
 * scaling, and with `postnorm` the residual is the normed input. This block
 * is NOT compatible with the fused residual-stream contract of the `TextModel`
 * template; a dedicated model loop must be used.
 */
class MiniMaxText01DecoderLayer : public infinicore::nn::Module {
public:
    MiniMaxText01DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                              size_t layer_idx,
                              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniMaxText01Attention, self_attn);
    INFINICORE_NN_MODULE(MiniMaxText01LinearAttention, linear_attn);
    INFINICORE_NN_MODULE(MiniMaxText01SparseMoeBlock, mlp);

private:
    size_t layer_idx_;
    int attn_type_;
    bool postnorm_;
    double layernorm_attention_alpha_;
    double layernorm_attention_beta_;
    double layernorm_mlp_alpha_;
    double layernorm_mlp_beta_;
};

} // namespace infinilm::models::minimax_text_01
