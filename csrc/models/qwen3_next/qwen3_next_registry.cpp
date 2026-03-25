#include "../models_registry.hpp"
#include "qwen3_next_for_causal_lm.hpp"

namespace {
struct Qwen3NextRegistry {
    Qwen3NextRegistry() {
        infinilm::models::register_causal_lm_model(
            "qwen3_next",
            [](std::shared_ptr<infinilm::config::ModelConfig> config, const infinicore::Device &device) {
                return std::make_shared<infinilm::models::qwen3_next::Qwen3NextForCausalLM>(config, device);
            });
        infinilm::models::register_model_config("qwen3_next", infinilm::models::qwen3_next::create_qwen3_next_model_config);
    }
} g_qwen3_next_registry;
} // namespace
