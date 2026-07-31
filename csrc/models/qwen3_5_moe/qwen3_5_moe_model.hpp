#pragma once

#include "../qwen3_5/qwen3_5_model.hpp"
#include "qwen3_5_moe_decoder_layer.hpp"

namespace infinilm::models::qwen3_5_moe {

using Qwen35MoeLanguageModel = infinilm::layers::causal_lm_templates::TextModel<Qwen35MoeDecoderLayer>;

class Qwen35MoeModel : public qwen3_5::Qwen35ModelBase {
public:
    Qwen35MoeModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device);

    infinicore::Tensor forward(const InfinilmModel::Input &input) const;

protected:
    INFINICORE_NN_MODULE(Qwen35MoeLanguageModel, language_model);
};

} // namespace infinilm::models::qwen3_5_moe
