#include "../models_registry.hpp"
#include "qwen3_vl_for_conditional_generation.hpp"

namespace {
struct Qwen3VlRegistry {
    Qwen3VlRegistry() {
        infinilm::models::register_causal_lm_model(
            "qwen3_vl",
            [](std::shared_ptr<infinilm::config::ModelConfig> config, const infinicore::Device &device) {
                return std::make_shared<infinilm::models::qwen3_vl::Qwen3VLForConditionalGeneration>(config, device);
            });
        infinilm::models::register_model_config("qwen3_vl", infinilm::models::qwen3_vl::create_qwen3_vl_model_config);
    }
} g_qwen3_vl_registry;
} // namespace
