#pragma once

#include "../../layers/causal_lm_templates/piecewise_text_causal_lm.hpp"
#include "../../layers/common_modules.hpp"
#include "minicpm5_moe_decoder_layer.hpp"
#include <memory>

namespace infinilm::models::minicpm5_moe {

using MiniCPM5MoeModel = infinilm::layers::causal_lm_templates::TextModel<MiniCPM5MoeDecoderLayer>;

using MiniCPM5MoeForCausalLM =
    infinilm::layers::causal_lm_templates::PiecewiseTextCausalLM<MiniCPM5MoeModel>;

std::shared_ptr<infinilm::config::ModelConfig>
create_minicpm5_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm5_moe
