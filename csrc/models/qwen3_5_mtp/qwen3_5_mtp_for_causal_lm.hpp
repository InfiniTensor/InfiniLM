#pragma once

#include "../../layers/common_modules.hpp"
#include "../../models/infinilm_model.hpp"
#include "../qwen3_5/qwen3_5_decoderLayer.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::qwen3_5_mtp {

class Qwen35MtpModel : public infinicore::nn::Module {
public:
    Qwen35MtpModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device);

    infinicore::Tensor embed_input_ids(const infinicore::Tensor &input_ids) const;

    infinicore::Tensor forward_with_hidden(const infinicore::Tensor &input_ids,
                                           const infinicore::Tensor &position_ids,
                                           const infinicore::Tensor &target_hidden_states) const;

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, pre_fc_norm_embedding);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, pre_fc_norm_hidden);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc);
    INFINICORE_NN_MODULE_VEC(infinilm::models::qwen3_5::Qwen35DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    infinicore::DataType dtype_;
    infinicore::Device device_;
    size_t hidden_size_;
};

class Qwen35MtpForCausalLM : public InfinilmModel {
public:
    Qwen35MtpForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    Output forward(const Input &input) const override;

protected:
    INFINICORE_NN_MODULE(Qwen35MtpModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_mtp_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5_mtp
