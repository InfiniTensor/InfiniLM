#pragma once

#include "../infinilm_model.hpp"
#include "../llama/llama_config.hpp"

namespace infinilm::models::minicpmv {

struct SiglipVisionConfig {
    size_t hidden_size = 1152;
    size_t intermediate_size = 4304;
    size_t num_hidden_layers = 27;
    size_t num_attention_heads = 16;
    size_t image_size = 980;
    size_t patch_size = 14;
    double layer_norm_eps = 1e-6;
    std::string hidden_act = "gelu_tanh";
    std::string model_type = "siglip";
};

struct MiniCPMVConfig : public InfinilmModel::Config {
    infinicore::DataType dtype = infinicore::DataType::BF16;
    std::string model_type = "minicpmv";

    llama::LlamaConfig llm_config;
    SiglipVisionConfig vision_config;

    size_t query_num = 64;
    bool drop_vision_last_layer = false;
    bool batch_vision_input = true;
};

} // namespace infinilm::models::minicpmv
