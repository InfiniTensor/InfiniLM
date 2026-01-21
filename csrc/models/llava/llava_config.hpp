#pragma once

#include "../infinilm_model.hpp"
#include "../llama/llama_config.hpp"

namespace infinilm::models::llava {

struct ClipVisionConfig {
    size_t hidden_size = 1024;
    size_t intermediate_size = 4096;
    size_t num_hidden_layers = 24;
    size_t num_attention_heads = 16;
    size_t image_size = 336;
    size_t patch_size = 14;
    double layer_norm_eps = 1e-5;
    std::string model_type = "clip_vision_model";
};

struct LlavaConfig : public InfinilmModel::Config {
    infinicore::DataType dtype = infinicore::DataType::F16;

    std::string model_type = "llava";

    int64_t image_token_index = 32000;
    int64_t pad_token_id = 0;
    int64_t vocab_size = 32064;
    int64_t ignore_index = -100;

    std::string projector_hidden_act = "gelu";
    int64_t vision_feature_layer = -2;
    std::string vision_feature_select_strategy = "default";

    bool tie_word_embeddings = false;

    ClipVisionConfig vision_config;
    llama::LlamaConfig text_config;
};

} // namespace infinilm::models::llava
