#include "minicpm_sala_model.hpp"

#include "infinicore/ops.hpp"
#include <cmath>
#include <stdexcept>

namespace infinilm::models::minicpm_sala {

MiniCPMSALAModel::MiniCPMSALAModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device,
                                   engine::distributed::RankInfo rank_info,
                                   backends::AttentionBackend attention_backend)
    : model_config_(std::move(model_config)),
      rank_info_(rank_info),
      attention_backend_(attention_backend) {

    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config_->get_dtype();
    compute_device_ = device;

    hidden_size_ = model_config_->get<size_t>("hidden_size");
    dim_model_base_ = model_config_->get_or<double>("dim_model_base", static_cast<double>(hidden_size_));
    scale_emb_ = model_config_->get_or<double>("scale_emb", 1.0);

    const size_t vocab_size = model_config_->get<size_t>("vocab_size");
    const size_t num_layers = model_config_->get<size_t>("num_hidden_layers");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size_, std::nullopt, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, model_config_->get<double>("rms_norm_eps"), dtype, device);

    // Shared rotary embedding (used by lightning layers only)
    INFINICORE_NN_MODULE_INIT(rotary_emb,
                              model_config_->get_head_dim(),
                              model_config_->get<size_t>("max_position_embeddings"),
                              model_config_->get<double>("rope_theta"),
                              infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype,
                              device,
                              model_config_->get_rope_scaling());

    // Mixer types per-layer decide attention flavor (minicpm4 vs lightning-attn).
    std::vector<std::string> mixer_types;
    try {
        mixer_types = model_config_->get<std::vector<std::string>>("mixer_types");
    } catch (...) {
        mixer_types.assign(num_layers, "minicpm4");
    }
    if (mixer_types.size() != num_layers) {
        mixer_types.resize(num_layers, mixer_types.empty() ? "minicpm4" : mixer_types.back());
    }

    layers_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.push_back(this->register_module<MiniCPMSALADecoderLayer>(
            "layers." + std::to_string(i), model_config_, device, i, mixer_types[i], rank_info_, attention_backend_));
        layers_.back()->set_rotary_emb(rotary_emb_);
    }
}

void MiniCPMSALAModel::reset_cache(const cache::CacheConfig *cache_config) {
    (void)cache_config;
    kv_cache_.reset();
    for (auto &layer : layers_) {
        layer->reset_cache();
    }
}

infinicore::Tensor MiniCPMSALAModel::forward(const infinicore::Tensor &input_ids,
                                             const infinicore::Tensor &position_ids,
                                             std::optional<infinicore::Tensor> past_sequence_lengths,
                                             std::optional<infinicore::Tensor> total_sequence_lengths,
                                             std::optional<infinicore::Tensor> input_offsets,
                                             std::optional<infinicore::Tensor> cu_seqlens,
                                             std::optional<infinicore::Tensor> block_tables,
                                             std::optional<infinicore::Tensor> slot_mapping) const {
    // MuP scaling baked into weights at load time for minicpm_sala; no forward scaling here.
    auto hs = embed_tokens_->forward(input_ids);

    for (size_t i = 0; i < layers_.size(); ++i) {
        hs = layers_[i]->forward(hs,
                                 position_ids,
                                 kv_cache_,
                                 past_sequence_lengths,
                                 total_sequence_lengths,
                                 input_offsets,
                                 cu_seqlens,
                                 block_tables,
                                 slot_mapping);
    }

    hs = norm_->forward(hs);
    return hs;
}

} // namespace infinilm::models::minicpm_sala

