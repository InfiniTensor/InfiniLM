#pragma once

#include "../../layers/common_modules.hpp"
#include "infinicore/nn/linear.hpp"
#include <cmath>
#include <memory>

namespace infinilm::models::fm9g {

// FM9GAttention: extends shared Attention, sets MuP alpha on o_proj
class FM9GAttention : public infinilm::layers::attention::Attention {
public:
    FM9GAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const infinicore::Device &device)
        : Attention(model_config, layer_idx, device) {
        float scale_depth = model_config->get_or<float>("scale_depth", 1.0f);
        if (scale_depth != 1.0f) {
            size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
            float scale_output = scale_depth / std::sqrt(static_cast<float>(num_hidden_layers));
            o_proj_->set_alpha(scale_output);
        }
    }
};

// FM9GMLP: extends shared MLP, sets MuP alpha on down_proj
class FM9GMLP : public infinilm::layers::mlp::MLP {
public:
    FM9GMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
            const infinicore::Device &device)
        : MLP(model_config, device) {
        float scale_depth = model_config->get_or<float>("scale_depth", 1.0f);
        if (scale_depth != 1.0f) {
            size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
            float scale_down = scale_depth / std::sqrt(static_cast<float>(num_hidden_layers));
            down_proj_->set_alpha(scale_down);
        }
    }
};

using FM9GDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<FM9GAttention, FM9GMLP>;

using FM9GModel = infinilm::layers::causal_lm_templates::TextModel<FM9GDecoderLayer>;

// FM9GForCausalLM: extends TextCausalLM, sets MuP alpha on lm_head
class FM9GForCausalLM : public infinilm::layers::causal_lm_templates::TextCausalLM<FM9GModel> {
public:
    FM9GForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device)
        : TextCausalLM<FM9GModel>(model_config, device) {
        if (model_config->get_config_json().contains("dim_model_base")) {
            float dim_model_base = model_config->get<float>("dim_model_base");
            size_t hidden_size = model_config->get<size_t>("hidden_size");
            float scale_lm_head = dim_model_base / static_cast<float>(hidden_size);
            this->lm_head_->set_alpha(scale_lm_head);
        }
    }
};

} // namespace infinilm::models::fm9g

namespace infinilm::models::fm9g {

std::shared_ptr<infinilm::config::ModelConfig> create_fm9g_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::fm9g
