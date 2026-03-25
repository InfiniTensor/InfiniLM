#include "../models_registry.hpp"
#include "minicpm_sala_for_causal_lm.hpp"

namespace {
struct MiniCPMSALARegistry {
    MiniCPMSALARegistry() {
        infinilm::models::register_causal_lm_model(
            "minicpm_sala",
            [](std::shared_ptr<infinilm::config::ModelConfig> config, const infinicore::Device &device) {
                return std::make_shared<infinilm::models::minicpm_sala::MiniCPMSALAForCausalLM>(config, device);
            });
        infinilm::models::register_model_config("minicpm_sala", infinilm::models::minicpm_sala::create_minicpm_sala_model_config);
    }
} g_minicpm_sala_registry;
} // namespace
