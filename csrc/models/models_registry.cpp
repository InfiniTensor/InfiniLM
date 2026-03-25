#include "models_registry.hpp"

#include <utility>

namespace infinilm::models {

namespace {

std::map<std::string, ModelCreator> &causal_lm_models_map() {
    static std::map<std::string, ModelCreator> m;
    return m;
}

std::map<std::string, ConfigCreator> &model_configs_map() {
    static std::map<std::string, ConfigCreator> m;
    return m;
}

} // namespace

void register_causal_lm_model(const std::string &model_type, ModelCreator creator) {
    causal_lm_models_map()[model_type] = std::move(creator);
}

void register_model_config(const std::string &model_type, ConfigCreator creator) {
    model_configs_map()[model_type] = std::move(creator);
}

const std::map<std::string, ModelCreator> &get_causal_lm_model_map() {
    return causal_lm_models_map();
}

const std::map<std::string, ConfigCreator> &get_model_config_map() {
    return model_configs_map();
}

} // namespace infinilm::models
