#pragma once

#include "../../layers/common_modules.hpp"
#include <cmath>
#include <exception>
#include <memory>

namespace infinilm::models::fm9g {

// FM9GAttention: extends shared Attention, sets MuP alpha on o_proj
class FM9GAttention : public infinilm::layers::attention::Attention {
public:
    FM9GAttention(infinilm::runtime::Runtime &runtime,
                  std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const engine::distributed::RankInfo &rank_info,
                  backends::AttentionBackend attention_backend,
                  const layers::moe::EPConfig &ep_config)
        : Attention(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config) {
        try {
            float scale_depth = model_config->get_or<float>("scale_depth", 1.0f);
            if (scale_depth != 1.0f) {
                size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
                float scale_output = scale_depth / std::sqrt(static_cast<float>(num_hidden_layers));
                o_proj_->set_alpha(scale_output);
            }
        } catch (...) {
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
            throw;
        }
    }
};

// FM9GMLP: extends shared MLP, sets MuP alpha on down_proj
class FM9GMLP : public infinilm::layers::mlp::MLP {
public:
    FM9GMLP(infinilm::runtime::Runtime &runtime,
            std::shared_ptr<infinilm::config::ModelConfig> model_config,
            size_t layer_idx,
            const engine::distributed::RankInfo &rank_info,
            backends::AttentionBackend attention_backend,
            const layers::moe::EPConfig &ep_config)
        : MLP(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config) {
        try {
            float scale_depth = model_config->get_or<float>("scale_depth", 1.0f);
            if (scale_depth != 1.0f) {
                size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
                float scale_down = scale_depth / std::sqrt(static_cast<float>(num_hidden_layers));
                down_proj_->set_alpha(scale_down);
            }
        } catch (...) {
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
            throw;
        }
    }
};

using FM9GDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<FM9GAttention, FM9GMLP>;

using FM9GModel = infinilm::layers::causal_lm_templates::TextModel<FM9GDecoderLayer, true>;

// FM9GForCausalLM: extends TextCausalLM, sets MuP alpha on lm_head
class FM9GForCausalLM : public infinilm::layers::causal_lm_templates::TextCausalLM<FM9GModel> {
public:
    FM9GForCausalLM(infinilm::runtime::Runtime &runtime,
                    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const engine::distributed::RankInfo &rank_info,
                    backends::AttentionBackend attention_backend,
                    const layers::moe::EPConfig &ep_config)
        : TextCausalLM<FM9GModel>(runtime, model_config, rank_info, attention_backend, ep_config) {
        try {
            if (model_config->get_config_json().contains("dim_model_base")) {
                float dim_model_base = model_config->get<float>("dim_model_base");
                size_t hidden_size = model_config->get<size_t>("hidden_size");
                float scale_lm_head = dim_model_base / static_cast<float>(hidden_size);
                this->set_lm_head_alpha(scale_lm_head);
            }
        } catch (...) {
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
            throw;
        }
    }
};

} // namespace infinilm::models::fm9g

namespace infinilm::models::fm9g {

std::shared_ptr<infinilm::config::ModelConfig> create_fm9g_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::fm9g
