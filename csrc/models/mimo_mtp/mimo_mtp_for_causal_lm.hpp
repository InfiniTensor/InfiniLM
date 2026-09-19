#pragma once

#include "../../layers/common_modules.hpp"
#include "../../models/infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <memory>

namespace infinilm::models::mimo_mtp {

/**
 * @brief MiMo MTP draft block.
 *
 * The published checkpoint stores one draft block per predicted token under
 * `model.mtp_layers.<depth>.`; the block fuses the normalized next-token
 * embedding with the normalized target hidden state, concatenating the hidden
 * state first, and runs one full-attention decoder layer before its final norm.
 * Position 0 carries no target hidden state, so the embedding stream is masked
 * there before the fusion norms.
 */
class MimoMtpModel : public infinicore::nn::Module {
public:
    MimoMtpModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device);

    infinicore::Tensor forward_with_hidden(const infinicore::Tensor &input_ids,
                                           const infinicore::Tensor &position_ids,
                                           const infinicore::Tensor &target_hidden_states) const;

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

protected:
    infinicore::Tensor embed_input_ids(const infinicore::Tensor &input_ids,
                                       const infinicore::Tensor &position_ids) const;

    infinicore::Tensor draft_layer(const infinicore::Tensor &position_ids,
                                   const infinicore::Tensor &hidden_states) const;

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, pre_fc_norm_embedding);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, pre_fc_norm_hidden);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::attention::Attention, self_attn);
    INFINICORE_NN_MODULE(infinilm::layers::MLP, mlp);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    infinicore::DataType dtype_;
    infinicore::Device device_;
    size_t hidden_size_;
};

class MimoMtpForCausalLM : public InfinilmModel {
public:
    MimoMtpForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device);

    Output forward(const Input &input) const override;

protected:
    INFINICORE_NN_MODULE(MimoMtpModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_mimo_mtp_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::mimo_mtp
