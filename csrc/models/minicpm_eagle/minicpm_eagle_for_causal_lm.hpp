#pragma once

#include "../../layers/common_modules.hpp"
#include "../../models/infinilm_model.hpp"
#include "../minicpm4/minicpm4_for_causal_lm.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::minicpm_eagle {

class MiniCPMEagleModel : public infinicore::nn::Module {
public:
    MiniCPMEagleModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device);

    infinicore::Tensor embed_input_ids(const infinicore::Tensor &input_ids) const;

    infinicore::Tensor forward_with_hidden(const infinicore::Tensor &input_ids,
                                           const infinicore::Tensor &position_ids,
                                           const infinicore::Tensor &target_hidden_states) const;

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_norm1);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_norm2);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc);
    INFINICORE_NN_MODULE_VEC(infinilm::models::minicpm4::MiniCPM4DecoderLayer, eagle_layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    infinicore::DataType dtype_;
    infinicore::Device device_;
    size_t hidden_size_;
};

class MiniCPMEagleForCausalLM : public InfinilmModel {
public:
    MiniCPMEagleForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device);

    Output forward(const Input &input) const override;

protected:
    INFINICORE_NN_MODULE(MiniCPMEagleModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_eagle_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm_eagle
