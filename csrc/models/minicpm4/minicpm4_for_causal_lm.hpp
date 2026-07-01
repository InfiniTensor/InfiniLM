#pragma once

#include "../../layers/common_modules.hpp"
#include "../../models/infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::minicpm4 {

class MiniCPM4Attention : public infinilm::layers::attention::Attention {
public:
    MiniCPM4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device);
};

class MiniCPM4MLP : public infinilm::layers::mlp::MLP {
public:
    MiniCPM4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                const infinicore::Device &device);
};

class MiniCPM4DecoderLayer : public infinicore::nn::Module {
public:
    MiniCPM4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

    void process_weights_after_loading() override {
        self_attn_->process_weights_after_loading();
        mlp_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        self_attn_->reset_runtime_state();
        mlp_->reset_runtime_state();
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(MiniCPM4Attention, self_attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniCPM4MLP, mlp);
};

class MiniCPM4Model : public infinicore::nn::Module {
public:
    MiniCPM4Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(MiniCPM4DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};

class MiniCPM4ForCausalLM : public InfinilmModel {
public:
    MiniCPM4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const infinicore::Device &device);

    Output forward(const Input &input) const override;

    infinicore::Tensor forward_hidden(const Input &input) const;

    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(MiniCPM4Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm4
