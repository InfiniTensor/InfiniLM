#include "../models_registry.hpp"
#include "qwen3_for_causal_lm.hpp"

namespace {
struct Qwen3Registry {
    Qwen3Registry() {
        infinilm::models::register_causal_lm_model(
            "qwen3",
            [](std::shared_ptr<infinilm::config::ModelConfig> config, const infinicore::Device &device) {
                return std::make_shared<infinilm::models::qwen3::Qwen3ForCausalLM>(config, device);
            });
        infinilm::models::register_model_config("qwen3", infinilm::models::qwen3::create_qwen3_model_config);
    }
} g_qwen3_registry;
} // namespace
