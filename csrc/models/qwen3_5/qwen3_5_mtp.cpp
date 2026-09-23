#include "qwen3_5_mtp.hpp"

#include "../../global_state/global_state.hpp"
#include <infinicore/ops/cat.hpp>
#include <stdexcept>

namespace infinilm::models::qwen3_5 {

Qwen35MTP::Qwen35MTP(std::shared_ptr<config::ModelConfig> config, const infinicore::Device &device) {
    if (config->get_or<size_t>("mtp_num_hidden_layers", 1) != 1
        || global_state::get_tensor_model_parallel_rank_info().pp_size != 1
        || config->get_or<bool>("mtp_use_dedicated_embeddings", false)
        || (config->get_config_json().contains("quantization_config")
            && !config->get_config_json()["quantization_config"].is_null()
            && !config->get_config_json()["quantization_config"].empty()
            && config->get_quant_scheme() != quantization::QuantScheme::FP8_BLOCK_W8A16)) {
        throw std::runtime_error("Qwen MTP requires one MTP layer, PP1, shared embeddings and BF16 or block FP8 weights.");
    }
    const auto hidden = config->get<size_t>("hidden_size");
    const auto eps = config->get<double>("rms_norm_eps");
    const auto dtype = config->get_dtype();
    INFINICORE_NN_MODULE_INIT(pre_fc_norm_embedding, hidden, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(pre_fc_norm_hidden, hidden, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc, 2 * hidden, hidden, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, hidden, eps, dtype, device);

    auto json = config->get_config_json();
    const size_t target_layers = config->get<size_t>("num_hidden_layers");
    json["layer_types"].push_back("full_attention");
    json["num_hidden_layers"] = target_layers + 1;
    auto mtp_config = std::make_shared<config::ModelConfig>(json);
    // Keep the checkpoint's `layers.0` name and a separate KV layer after target.
    layer_ = register_module<Qwen35DecoderLayer>("layers.0", mtp_config, target_layers, device);
}

infinicore::Tensor Qwen35MTP::forward(const infinicore::Tensor &embeddings,
                                      const infinicore::Tensor &target_hidden,
                                      const infinicore::Tensor &positions) const {
    if (embeddings->shape() != target_hidden->shape()
        || embeddings->dtype() != target_hidden->dtype()) {
        throw std::runtime_error("Qwen MTP requires matching embedding and target hidden-state shapes and dtypes.");
    }
    auto embed = pre_fc_norm_embedding_->forward(embeddings);
    auto hidden = pre_fc_norm_hidden_->forward(target_hidden);
    auto fused = infinicore::op::cat({embed, hidden}, -1);
    hidden = fc_->forward(fused);
    hidden = layer_->forward(positions, hidden);
    return norm_->forward(hidden);
}

} // namespace infinilm::models::qwen3_5
