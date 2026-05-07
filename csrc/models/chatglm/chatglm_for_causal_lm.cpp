#include "chatglm_for_causal_lm.hpp"
#include "../llama/llama_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::chatglm {

std::shared_ptr<infinilm::config::ModelConfig> create_chatglm_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("chatglm" != model_type) {
        throw std::runtime_error(
            "infinilm::models::chatglm::create_chatglm_model_config: model_type is not chatglm");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    auto rename_key = [&config_json](const std::string &old_key, const std::string &new_key) {
        if (config_json.contains(old_key) && !config_json.contains(new_key)) {
            config_json[new_key] = config_json[old_key];
        }
    };

    rename_key("num_layers", "num_hidden_layers");
    rename_key("multi_query_group_num", "num_key_value_heads");
    rename_key("kv_channels", "head_dim");
    rename_key("layernorm_epsilon", "rms_norm_eps");
    rename_key("seq_length", "max_position_embeddings");
    rename_key("ffn_hidden_size", "intermediate_size");

    if (!config_json.contains("vocab_size") && config_json.contains("padded_vocab_size")) {
        config_json["vocab_size"] = config_json["padded_vocab_size"];
    }

    if (!config_json.contains("attention_bias")) { config_json["attention_bias"] = true; }

    if (!config_json.contains("rope_theta")) {
        config_json["rope_theta"] = 10000.0;
    }

    return model_config;
}

} // namespace infinilm::models::chatglm

namespace {

#ifndef USE_CLASSIC_LLAMA

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    chatglm,
    infinilm::models::llama::LlamaForCausalLM,
    infinilm::models::chatglm::create_chatglm_model_config);

#endif

} // namespace
