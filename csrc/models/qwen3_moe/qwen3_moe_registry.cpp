#include "../models_registry.hpp"
#include "qwen3_moe_for_causal_lm.hpp"

namespace {
struct Qwen3MoeRegistry {
    Qwen3MoeRegistry() {
        infinilm::models::register_causal_lm_model(
            "qwen3_moe",
            [](std::shared_ptr<infinilm::config::ModelConfig> config, const infinicore::Device &device) {
                return std::make_shared<infinilm::models::qwen3_moe::Qwen3MoeForCausalLM>(config, device);
            });
        infinilm::models::register_model_config("qwen3_moe", infinilm::models::qwen3_moe::create_qwen3_moe_model_config);
    }
} g_qwen3_moe_registry;
} // namespace
