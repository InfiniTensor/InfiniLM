#pragma once

#include "../../layers/TemplateCausalLM.hpp"
#include "../../layers/TemplateModel.hpp"
#include "../../layers/attention.hpp"
#include "../../layers/mlp.hpp"
#include "minicpm_sala_attention.hpp"
#include "minicpm_sala_decoderLayer.hpp"
#include <memory>
#include <variant>

namespace infinilm::models::minicpm_sala {

/** @brief Type alias for MiniCPM-SALA MLP module */
using MiniCPMMLP = infinilm::models::layers::MLP;

/** @brief MiniCPM-SALA attention variant (InfLLMv2 / Lightning) */
using MiniCPMSALAAttention = std::variant<std::shared_ptr<InfLLMv2Attention>, std::shared_ptr<LightningAttention>>;

/** @brief MiniCPM-SALA decoder layer type alias */
using MiniCPMSALADecoderLayerImpl = MiniCPMSALADecoderLayer<MiniCPMSALAAttention, MiniCPMMLP>;

/** @brief MiniCPM-SALA model architecture (without language modeling head) */
using MiniCPMSALAModel = infinilm::models::layers::TemplateModel<MiniCPMSALADecoderLayerImpl>;

/** @brief MiniCPM model for Causal Language Modeling */
using MiniCPMSALAForCausalLM = infinilm::models::layers::TemplateCausalLM<MiniCPMSALAModel>;

} // namespace infinilm::models::minicpm_sala
