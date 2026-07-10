#include "ernie4_5_vision.hpp"

#include "infinicore/ops.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::ernie4_5_vl {
Ernie45VisionLayerNorm::Ernie45VisionLayerNorm(size_t normalized_shape,
                                               double eps,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device)
    : eps_(eps) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({normalized_shape}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({normalized_shape}, dtype, device));
}

infinicore::Tensor Ernie45VisionLayerNorm::forward(const infinicore::Tensor &hidden_states) const {
    return infinicore::op::layer_norm(hidden_states, weight_, bias_, static_cast<float>(eps_));
}

Ernie45VisionPatchEmbed::Ernie45VisionPatchEmbed(const nlohmann::json &config,
                                                 const infinicore::DataType &dtype,
                                                 const infinicore::Device &device) {
    const size_t in_channels = config.value("in_channels", config.value("in_chans", 3));
    const size_t patch_size = config.value("patch_size", config.value("spatial_patch_size", 14));
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    INFINICORE_NN_MODULE_INIT(proj, in_channels * patch_size * patch_size, embed_dim, false, dtype, device);
}

infinicore::Tensor Ernie45VisionPatchEmbed::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    return proj_->forward(hidden_states_mutable);
}

Ernie45VisionAttention::Ernie45VisionAttention(const nlohmann::json &config,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device) {
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    INFINICORE_NN_MODULE_INIT(qkv, embed_dim, embed_dim * 3, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, embed_dim, embed_dim, true, dtype, device);
}

infinicore::Tensor Ernie45VisionAttention::forward(const infinicore::Tensor &) const {
    throw std::runtime_error("Ernie45VisionAttention::forward requires variable-length vision RoPE attention; operator wiring is not implemented yet");
}

Ernie45VisionMLP::Ernie45VisionMLP(const nlohmann::json &config,
                                   const infinicore::DataType &dtype,
                                   const infinicore::Device &device) {
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    const size_t hidden_dim = static_cast<size_t>(static_cast<double>(embed_dim) * config.value("mlp_ratio", 4.0));
    INFINICORE_NN_MODULE_INIT(fc1, embed_dim, hidden_dim, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc2, hidden_dim, embed_dim, true, dtype, device);
}

infinicore::Tensor Ernie45VisionMLP::forward(const infinicore::Tensor &) const {
    throw std::runtime_error("Ernie45VisionMLP::forward requires quick_gelu/GELU activation dispatch; operator wiring is not implemented yet");
}

Ernie45VisionBlock::Ernie45VisionBlock(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    INFINICORE_NN_MODULE_INIT(norm1, embed_dim, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, embed_dim, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, dtype, device);
}

infinicore::Tensor Ernie45VisionBlock::forward(const infinicore::Tensor &) const {
    throw std::runtime_error("Ernie45VisionBlock::forward is not implemented yet");
}

Ernie45VisionModel::Ernie45VisionModel(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    const size_t depth = config.value("depth", 32);
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    INFINICORE_NN_MODULE_INIT(patch_embed, config, dtype, device);
    blocks_.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        blocks_.push_back(this->register_module<Ernie45VisionBlock>("blocks." + std::to_string(i), config, dtype, device));
    }
    INFINICORE_NN_MODULE_INIT(ln, embed_dim, 1e-6, dtype, device);
}

infinicore::Tensor Ernie45VisionModel::forward(const infinicore::Tensor &) const {
    throw std::runtime_error("Ernie45VisionModel::forward requires grid_thw-aware DFN RoPE vision attention; operator wiring is not implemented yet");
}

Ernie45ResamplerModel::Ernie45ResamplerModel(const nlohmann::json &config,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device) {
    const size_t pixel_hidden_size = config.value("pixel_hidden_size", 1280);
    const size_t hidden_size = config.value("hidden_size", 2560);
    const size_t spatial_conv_size = config.value("spatial_conv_size", 2);
    const size_t temporal_conv_size = config.value("temporal_conv_size", 2);
    const size_t spatial_dim = pixel_hidden_size * spatial_conv_size * spatial_conv_size;
    const size_t temporal_dim = spatial_dim * temporal_conv_size;
    use_temporal_conv_ = config.value("use_temporal_conv", true);

    spatial_linear_0_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("spatial_linear.0", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear_2_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("spatial_linear.2", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear_3_ = this->register_module<Ernie45VisionLayerNorm>("spatial_linear.3", spatial_dim, 1e-6, dtype, device);
    if (use_temporal_conv_) {
        temporal_linear_0_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("temporal_linear.0", temporal_dim, spatial_dim, true, dtype, device);
        temporal_linear_2_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("temporal_linear.2", spatial_dim, spatial_dim, true, dtype, device);
        temporal_linear_3_ = this->register_module<Ernie45VisionLayerNorm>("temporal_linear.3", spatial_dim, 1e-6, dtype, device);
    }
    INFINICORE_NN_MODULE_INIT(mlp, spatial_dim, hidden_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(after_norm, hidden_size, config.value("rms_norm_eps", 1e-5), dtype, device);
}

infinicore::Tensor Ernie45ResamplerModel::forward(const infinicore::Tensor &) const {
    throw std::runtime_error("Ernie45ResamplerModel::forward requires spatial/temporal variable-resolution token packing; operator wiring is not implemented yet");
}

} // namespace infinilm::models::ernie4_5_vl



