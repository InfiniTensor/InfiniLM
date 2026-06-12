#include "rotary_embedding_factory.hpp"
#include "../../config/model_config.hpp"

namespace infinilm::layers::rotary_embedding {

std::unordered_map<std::string, ScalingCreator> &get_scaling_registry() {
    static std::unordered_map<std::string, ScalingCreator> registry;
    return registry;
}

std::shared_ptr<infinicore::nn::RopeScalingConfig>
make_scaling_config(const std::shared_ptr<config::ModelConfig> &model_config) {
    if (!model_config || !model_config->get_config_json().contains("rope_scaling") || model_config->get_config_json()["rope_scaling"].is_null()) {
        return nullptr;
    }

    const auto &rope_scaling = model_config->get_config_json()["rope_scaling"];
    if (!rope_scaling.is_object()) {
        throw std::runtime_error("rope_scaling must be an object");
    }

    std::string scaling_type;
    if (rope_scaling.contains("type")) {
        scaling_type = rope_scaling["type"].get<std::string>();
    } else if (rope_scaling.contains("rope_type")) {
        scaling_type = rope_scaling["rope_type"].get<std::string>();
    } else {
        throw std::runtime_error("rope_scaling must contain 'type' or 'rope_type' field");
    }

    // Registry routing: delegate construction to the specific creator
    auto &registry = get_scaling_registry();
    auto it = registry.find(scaling_type);
    if (it != registry.end()) {
        return it->second(model_config);
    }

    throw std::runtime_error("Unsupported rope_scaling_type: " + scaling_type);
}

} // namespace infinilm::layers::rotary_embedding
