#include "qwen3_5_vision.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_5 {
namespace {

size_t get_size_or_first(const nlohmann::json &config, const char *key, size_t default_value) {
    if (!config.contains(key) || config.at(key).is_null()) {
        return default_value;
    }
    const auto &value = config.at(key);
    if (value.is_array()) {
        return value.empty() ? default_value : value.at(0).get<size_t>();
    }
    return value.get<size_t>();
}

} // namespace

Qwen35VisionPatchProj::Qwen35VisionPatchProj(size_t in_channels,
                                             size_t hidden_size,
                                             size_t temporal_patch_size,
                                             size_t patch_size,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device)
    : in_channels_(in_channels),
      hidden_size_(hidden_size),
      temporal_patch_size_(temporal_patch_size),
      patch_size_(patch_size) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({hidden_size_, in_channels_, temporal_patch_size_, patch_size_, patch_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({hidden_size_}, dtype, device));
}

infinicore::Tensor Qwen35VisionPatchProj::forward(const infinicore::Tensor &hidden_states) const {
    throw std::runtime_error("Qwen35VisionPatchProj::forward is not implemented yet");
}

Qwen35VisionPatchEmbed::Qwen35VisionPatchEmbed(const nlohmann::json &config,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device) {
    const size_t in_channels = config.value("in_channels", 3);
    const size_t hidden_size = config.value("hidden_size", 1152);
    const size_t temporal_patch_size = get_size_or_first(config, "temporal_patch_size", 2);
    const size_t patch_size = get_size_or_first(config, "patch_size", 16);
    INFINICORE_NN_MODULE_INIT(proj, in_channels, hidden_size, temporal_patch_size, patch_size, dtype, device);
}

infinicore::Tensor Qwen35VisionPatchEmbed::forward(const infinicore::Tensor &hidden_states) const {
    return proj_->forward(hidden_states);
}

Qwen35VisionAttention::Qwen35VisionAttention(const nlohmann::json &config,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device)
    : hidden_size_(config.value("hidden_size", 1152)),
      num_heads_(config.value("num_heads", 16)) {
    INFINICORE_NN_MODULE_INIT(qkv, hidden_size_, hidden_size_ * 3, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, hidden_size_, hidden_size_, true, dtype, device);
}

infinicore::Tensor Qwen35VisionAttention::forward(const infinicore::Tensor &hidden_states) const {
    throw std::runtime_error("Qwen35VisionAttention::forward is not implemented yet");
}

Qwen35VisionMLP::Qwen35VisionMLP(const nlohmann::json &config,
                                 const infinicore::DataType &dtype,
                                 const infinicore::Device &device) {
    const size_t hidden_size = config.value("hidden_size", 1152);
    const size_t intermediate_size = config.value("intermediate_size", 4304);
    INFINICORE_NN_MODULE_INIT(linear_fc1, hidden_size, intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc2, intermediate_size, hidden_size, true, dtype, device);
}

infinicore::Tensor Qwen35VisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    throw std::runtime_error("Qwen35VisionMLP::forward is not implemented yet");
}

Qwen35VisionBlock::Qwen35VisionBlock(const nlohmann::json &config,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device) {
    const size_t hidden_size = config.value("hidden_size", 1152);
    const double norm_eps = config.value("layer_norm_eps", config.value("rms_norm_eps", 1e-6));
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, dtype, device);
}

infinicore::Tensor Qwen35VisionBlock::forward(const infinicore::Tensor &hidden_states) const {
    throw std::runtime_error("Qwen35VisionBlock::forward is not implemented yet");
}

Qwen35VisionPatchMerger::Qwen35VisionPatchMerger(const nlohmann::json &config,
                                                 const infinicore::DataType &dtype,
                                                 const infinicore::Device &device) {
    const size_t hidden_size = config.value("hidden_size", 1152);
    const size_t out_hidden_size = config.value("out_hidden_size", hidden_size);
    const size_t spatial_merge_size = config.value("spatial_merge_size", 2);
    const size_t merged_size = hidden_size * spatial_merge_size * spatial_merge_size;
    const double norm_eps = config.value("layer_norm_eps", config.value("rms_norm_eps", 1e-6));
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc1, merged_size, merged_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc2, merged_size, out_hidden_size, true, dtype, device);
}

infinicore::Tensor Qwen35VisionPatchMerger::forward(const infinicore::Tensor &hidden_states) const {
    throw std::runtime_error("Qwen35VisionPatchMerger::forward is not implemented yet");
}

Qwen35VisionModel::Qwen35VisionModel(const nlohmann::json &config,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device) {
    const size_t hidden_size = config.value("hidden_size", 1152);
    const size_t num_position_embeddings = config.value("num_position_embeddings", 2304);
    const size_t depth = config.value("depth", config.value("num_hidden_layers", 27));

    INFINICORE_NN_MODULE_INIT(patch_embed, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(pos_embed, num_position_embeddings, hidden_size, std::nullopt, dtype, device);
    blocks_.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        blocks_.push_back(this->register_module<Qwen35VisionBlock>("blocks." + std::to_string(i), config, dtype, device));
    }
    INFINICORE_NN_MODULE_INIT(merger, config, dtype, device);
}

infinicore::Tensor Qwen35VisionModel::forward(const infinicore::Tensor &pixel_values) const {
    throw std::runtime_error("Qwen35VisionModel::forward is not implemented yet");
}

} // namespace infinilm::models::qwen3_5
