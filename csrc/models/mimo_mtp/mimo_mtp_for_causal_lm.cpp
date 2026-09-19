#include "mimo_mtp_for_causal_lm.hpp"
#include "../models_registry.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::mimo_mtp {

namespace {

/// Positions of the draft batch, read back from the device.
///
/// The block masks the embedding of every position 0 row, which is a handful
/// of scalars; the host round-trip follows the one the static attention
/// backend already uses for its sequence lengths.
std::vector<int64_t> host_positions(const infinicore::Tensor &position_ids) {
    auto host = position_ids->to(infinicore::Device::cpu());
    const size_t numel = host->numel();
    std::vector<int64_t> values;
    values.reserve(numel);
    if (host->dtype() == infinicore::DataType::I32) {
        const auto *data = reinterpret_cast<const int32_t *>(host->data());
        values.assign(data, data + numel);
    } else if (host->dtype() == infinicore::DataType::I64) {
        const auto *data = reinterpret_cast<const int64_t *>(host->data());
        values.assign(data, data + numel);
    } else {
        throw std::runtime_error(
            "infinilm::models::mimo_mtp: position_ids must be int32 or int64");
    }
    return values;
}

} // namespace

namespace {

/// A positive whole number of draft layers; any other JSON value is a config
/// defect rather than a depth to fall back from. Negative and zero counts are
/// rejected here so the diagnostic reports the value that was written instead
/// of its unsigned wraparound.
size_t whole_number(const nlohmann::json &value, const std::string &key) {
    if (value.is_number_unsigned()) {
        const auto number = value.get<uint64_t>();
        if (number >= 1 && number <= std::numeric_limits<size_t>::max()) {
            return static_cast<size_t>(number);
        }
    } else if (value.is_number_integer()) {
        const auto number = value.get<int64_t>();
        if (number >= 1) {
            return static_cast<size_t>(number);
        }
    }
    throw std::runtime_error(
        "infinilm::models::mimo_mtp: " + key + " must be a positive whole number of draft layers, got "
        + value.dump());
}

/// Published draft depth of the checkpoint, as its draft config states it.
size_t published_depth(const nlohmann::json &json) {
    if (json.contains("text_config") && json["text_config"].is_object()) {
        const auto &text_config = json["text_config"];
        if (text_config.contains("num_hidden_layers")) {
            return whole_number(text_config["num_hidden_layers"], "text_config.num_hidden_layers");
        }
    }
    if (json.contains("num_nextn_predict_layers")) {
        return whole_number(json["num_nextn_predict_layers"], "num_nextn_predict_layers");
    }
    return 1;
}

} // namespace

MimoMtpModel::MimoMtpModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device)
    : dtype_(model_config->get_dtype()),
      device_(device),
      hidden_size_(model_config->get<size_t>("hidden_size")) {
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    // The draft holds its own copy of the target's embedding table and head.
    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size_, std::nullopt, dtype_, device_);
    // Fusion: the two streams are normalized separately, concatenated with the
    // target hidden state first, and projected back to hidden_size.
    INFINICORE_NN_MODULE_INIT(pre_fc_norm_embedding, hidden_size_, rms_norm_eps, dtype_, device_);
    INFINICORE_NN_MODULE_INIT(pre_fc_norm_hidden, hidden_size_, rms_norm_eps, dtype_, device_);
    INFINICORE_NN_MODULE_INIT(fc, hidden_size_ * 2, hidden_size_, false, dtype_, device_);

    // One decoder layer, inlined here: the published layout keeps the layer's
    // tensors directly under the draft block instead of nesting them.
    const size_t layer_idx = 0;
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size_, rms_norm_eps, dtype_, device_);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size_, rms_norm_eps, dtype_, device_);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device_);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device_);
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, rms_norm_eps, dtype_, device_);
}

infinicore::Tensor MimoMtpModel::embed_input_ids(const infinicore::Tensor &input_ids,
                                                 const infinicore::Tensor &position_ids) const {
    auto input_embeds = embed_tokens_->forward(input_ids);
    const auto positions = host_positions(position_ids);
    const auto shape = input_embeds->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    // Position 0 has no target hidden state to fuse with, so its embedding is
    // dropped before the fusion norms. Positions are either one value per
    // token or one row shared by the batch.
    const size_t row_major = batch_size * seq_len;
    if (positions.size() != row_major && positions.size() != seq_len) {
        throw std::runtime_error(
            "infinilm::models::mimo_mtp: position_ids do not match the draft input shape");
    }
    const bool per_token = positions.size() == row_major;
    infinicore::Tensor zeros;
    for (size_t row = 0; row < batch_size; ++row) {
        for (size_t col = 0; col < seq_len; ++col) {
            const size_t index = per_token ? row * seq_len + col : col;
            if (positions[index] != 0) {
                continue;
            }
            if (zeros.empty()) {
                zeros = infinicore::Tensor::zeros({1, 1, hidden_size_}, dtype_, device_);
            }
            input_embeds->narrow({{0, row, 1}, {1, col, 1}})->copy_from(zeros);
        }
    }
    return input_embeds;
}

infinicore::Tensor MimoMtpModel::forward_with_hidden(const infinicore::Tensor &input_ids,
                                                     const infinicore::Tensor &position_ids,
                                                     const infinicore::Tensor &target_hidden_states) const {
    auto input_embeds = pre_fc_norm_embedding_->forward(embed_input_ids(input_ids, position_ids));
    auto target_hidden = pre_fc_norm_hidden_->forward(target_hidden_states);
    auto fused_shape = input_embeds->shape();
    fused_shape.back() = hidden_size_ * 2;
    auto fused_input = infinicore::Tensor::empty(fused_shape, input_embeds->dtype(), input_embeds->device());
    fused_input->narrow({{fused_shape.size() - 1, 0, hidden_size_}})->copy_from(target_hidden);
    fused_input->narrow({{fused_shape.size() - 1, hidden_size_, hidden_size_}})->copy_from(input_embeds);
    auto hidden_states = fc_->forward(fused_input);

    hidden_states = draft_layer(position_ids, hidden_states);

    // The checkpoint carries final_layernorm: the draft hidden state is
    // normalized before the head and is the tensor the next draft step consumes
    // as its previous hidden state.
    return norm_->forward(hidden_states);
}

infinicore::Tensor MimoMtpModel::draft_layer(const infinicore::Tensor &position_ids,
                                             const infinicore::Tensor &hidden_states) const {
    auto residual = hidden_states;
    auto normed = input_layernorm_->forward(hidden_states);
    normed = self_attn_->forward(position_ids, normed);
    normed = infinicore::op::add(residual, normed);

    residual = normed;
    normed = post_attention_layernorm_->forward(normed);
    normed = mlp_->forward(normed);
    return infinicore::op::add(residual, normed);
}

infinicore::Tensor MimoMtpModel::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto zero_hidden_states = infinicore::Tensor::zeros({input_ids->shape()[0], input_ids->shape()[1], hidden_size_}, dtype_, device_);
    return forward_with_hidden(input_ids, positions, zero_hidden_states);
}

MimoMtpForCausalLM::MimoMtpForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output MimoMtpForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    infinicore::Tensor hidden_states;
    if (input.target_hidden_states.has_value()) {
        hidden_states = model_->forward_with_hidden(input.input_ids.value(), input.position_ids.value(), input.target_hidden_states.value());
    } else {
        hidden_states = model_->forward(input);
    }
    auto logits = lm_head_->forward(hidden_states);
    return {logits, hidden_states};
}

std::shared_ptr<infinilm::config::ModelConfig> create_mimo_mtp_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("mimo_mtp" != model_type) {
        throw std::runtime_error("infinilm::models::mimo_mtp::create_mimo_mtp_model_config: model_type is not mimo_mtp");
    }

    auto &json = model_config->get_config_json();
    // The draft block composes one published depth, and checkpoints that publish
    // more than one give each depth its own weights. The depth is read where the
    // draft config states it: a standalone draft config states it directly,
    // while a config materialised from a checkpoint that embeds its head carries
    // the depth its weights publish next to a top level describing the target.
    const size_t num_hidden_layers = published_depth(json);
    if (num_hidden_layers != 1) {
        throw std::runtime_error(
            "infinilm::models::mimo_mtp: this draft block composes one published "
            "depth, but the draft config publishes "
            + std::to_string(num_hidden_layers) + " (num_nextn_predict_layers="
            + (json.contains("num_nextn_predict_layers")
                   ? json["num_nextn_predict_layers"].dump()
                   : std::string("absent"))
            + ")");
    }
    json["num_hidden_layers"] = num_hidden_layers;

    return model_config;
}

} // namespace infinilm::models::mimo_mtp

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    mimo_mtp,
    infinilm::models::mimo_mtp::MimoMtpForCausalLM,
    infinilm::models::mimo_mtp::create_mimo_mtp_model_config);
} // namespace
