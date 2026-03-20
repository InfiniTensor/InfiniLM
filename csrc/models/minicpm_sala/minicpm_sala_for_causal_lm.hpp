#pragma once

#include "../../layers/common_modules.hpp"
#include "minicpm_sala_attention.hpp"
#include "minicpm_sala_decoderLayer.hpp"
#include <memory>
#include <variant>

namespace infinilm::models::minicpm_sala {

/** @brief Type alias for MiniCPM-SALA MLP module */
using MiniCPMMLP = infinilm::layers::MLP;

/** @brief MiniCPM-SALA attention variant (InfLLMv2 / Lightning) */
using MiniCPMSALAAttention = std::variant<std::shared_ptr<InfLLMv2Attention>, std::shared_ptr<LightningAttention>>;

/** @brief MiniCPM-SALA decoder layer type alias */
using MiniCPMSALADecoderLayerImpl = MiniCPMSALADecoderLayer<MiniCPMSALAAttention, MiniCPMMLP>;

/** @brief MiniCPM-SALA model architecture (without language modeling head) */
using MiniCPMSALAModel = infinilm::layers::TextModel<MiniCPMSALADecoderLayerImpl>;

/** @brief MiniCPM model for Causal Language Modeling */
using MiniCPMSALAForCausalLM = infinilm::layers::TextCausalLM<MiniCPMSALAModel>;

static std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_sala_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("minicpm_sala" != model_type) {
        throw std::runtime_error("create_minicpm_sala_model_config: model_type is not minicpm_sala");
    }
    return model_config;
}

} // namespace infinilm::models::minicpm_sala
