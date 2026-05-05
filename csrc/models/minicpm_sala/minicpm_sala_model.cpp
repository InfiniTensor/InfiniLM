#include "minicpm_sala_model.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace infinilm::models::minicpm_sala {

MiniCPMSALAModel::MiniCPMSALAModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device) {

    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config->get_dtype();

    hidden_size_ = model_config->get<size_t>("hidden_size");

    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size_, std::nullopt, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, model_config->get<double>("rms_norm_eps"), dtype, device);

    // Mixer types per-layer decide attention flavor (minicpm4 vs lightning-attn).
    std::vector<std::string> mixer_types;
    try {
        mixer_types = model_config->get<std::vector<std::string>>("mixer_types");
    } catch (...) {
        mixer_types.assign(num_layers, "minicpm4");
    }
    if (mixer_types.size() != num_layers) {
        mixer_types.resize(num_layers, mixer_types.empty() ? "minicpm4" : mixer_types.back());
    }

    layers_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.push_back(this->register_module<MiniCPMSALADecoderLayer>(
            "layers." + std::to_string(i), model_config, device, i, mixer_types[i]));
    }
}

void MiniCPMSALAModel::reset_state() {
    for (auto &layer : layers_) {
        layer->reset_attn_state();
    }
}

infinicore::Tensor MiniCPMSALAModel::forward(const infinicore::Tensor &input_ids,
                                             const infinicore::Tensor &position_ids) const {
    // MuP scaling baked into weights at load time for minicpm_sala; no forward scaling here.
    auto hs = embed_tokens_->forward(input_ids);

    for (size_t i = 0; i < layers_.size(); ++i) {
        hs = layers_[i]->forward(hs, position_ids);
    }

    hs = norm_->forward(hs);
    return hs;
}

} // namespace infinilm::models::minicpm_sala
