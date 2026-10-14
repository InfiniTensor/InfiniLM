#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../../layers/causal_lm_templates/text_model.hpp"
#include "minicpm5_moe_decoder_layer.hpp"

#include <memory>

namespace infinilm::models::minicpm5_moe {

using MiniCPM5MoEModel = infinilm::layers::causal_lm_templates::TextModel<MiniCPM5MoEDecoderLayer>;
using MiniCPM5MoEForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<MiniCPM5MoEModel>;

std::shared_ptr<infinilm::config::ModelConfig>
create_minicpm5_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm5_moe
