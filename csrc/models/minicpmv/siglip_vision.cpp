#include "siglip_vision.hpp"
#include "backends/operators/operators.hpp"


#include <cmath>
#include <cstring>
#include <stdexcept>

namespace infinilm::models::minicpmv {

SiglipPatchEmbedding::SiglipPatchEmbedding(size_t in_channels,
                                           size_t out_channels,
                                           size_t patch_size,
                                           const infinicore::DataType &dtype,
                                           const infinicore::Device &device)
    : patch_size_(patch_size) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_channels, in_channels, patch_size_, patch_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({out_channels}, dtype, device));
}

infinicore::Tensor SiglipPatchEmbedding::forward(const infinicore::Tensor &pixel_values) const {
    std::vector<infinicore::Size> pads{0, 0};
    std::vector<infinicore::Size> strides{patch_size_, patch_size_};
    std::vector<infinicore::Size> dilations{1, 1};
    return infinilm::backends::ops::conv2d(pixel_values, weight_, bias_, pads, strides, dilations);
}

SiglipVisionEmbeddings::SiglipVisionEmbeddings(const nlohmann::json &config,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device)
    : hidden_size_(config.value("hidden_size", 768)),
      patch_size_(config.value("patch_size", 16)),
      num_positions_((config.value("image_size", 224) / config.value("patch_size", 16)) * (config.value("image_size", 224) / config.value("patch_size", 16))) {
    INFINICORE_NN_MODULE_INIT(patch_embedding, 3, hidden_size_, patch_size_, dtype, device);
    INFINICORE_NN_MODULE_INIT(position_embedding, num_positions_, hidden_size_, std::nullopt, dtype, device);
}

infinicore::Tensor SiglipVisionEmbeddings::forward(const infinicore::Tensor &pixel_values,
                                                   const std::optional<infinicore::Tensor> &tgt_sizes) const {
    auto patch_embeds = patch_embedding_->forward(pixel_values);
    auto batch_size = patch_embeds->size(0);
    auto seq_len = patch_embeds->size(2) * patch_embeds->size(3);

    auto embeddings = patch_embeds->view({batch_size, hidden_size_, seq_len})->permute({0, 2, 1});

    // Build position ids on CPU
    auto pos_ids_cpu = infinicore::Tensor::zeros({batch_size, seq_len}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *pos_ptr = reinterpret_cast<int64_t *>(pos_ids_cpu->data());

    const size_t num_patches_per_side = static_cast<size_t>(std::sqrt(static_cast<double>(num_positions_)));

    std::vector<int64_t> tgt_sizes_host;
    if (tgt_sizes.has_value()) {
        auto tgt_cpu = tgt_sizes.value()->to(infinicore::Device::cpu());
        auto n = tgt_cpu->numel();
        tgt_sizes_host.resize(n);
        std::memcpy(tgt_sizes_host.data(), tgt_cpu->data(), n * sizeof(int64_t));
    }

    for (size_t b = 0; b < batch_size; ++b) {
        size_t nb_h = num_patches_per_side;
        size_t nb_w = num_patches_per_side;
        if (!tgt_sizes_host.empty()) {
            nb_h = static_cast<size_t>(tgt_sizes_host[b * 2]);
            nb_w = static_cast<size_t>(tgt_sizes_host[b * 2 + 1]);
        }
        size_t patch_len = nb_h * nb_w;
        for (size_t ih = 0; ih < nb_h; ++ih) {
            size_t bh = (ih * num_patches_per_side) / nb_h;
            for (size_t iw = 0; iw < nb_w; ++iw) {
                size_t bw = (iw * num_patches_per_side) / nb_w;
                size_t pos_id = bh * num_patches_per_side + bw;
                size_t idx = ih * nb_w + iw;
                if (idx < seq_len) {
                    pos_ptr[b * seq_len + idx] = static_cast<int64_t>(pos_id);
                }
            }
        }
    }

    auto pos_ids = pos_ids_cpu->to(embeddings->device());
    auto pos_embed = position_embedding_->forward(pos_ids);
    return infinilm::backends::ops::add(embeddings, pos_embed);
}

SiglipAttention::SiglipAttention(const nlohmann::json &config,
                                 const infinicore::DataType &dtype,
                                 const infinicore::Device &device)
    : embed_dim_(config.value("hidden_size", 768)),
      num_heads_(config.value("num_attention_heads", 12)),
      head_dim_(config.value("hidden_size", 768) / config.value("num_attention_heads", 12)),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (embed_dim_ % num_heads_ != 0) {
        throw std::runtime_error("SiglipAttention: embed_dim must be divisible by num_heads");
    }
    INFINICORE_NN_MODULE_INIT(q_proj, embed_dim_, embed_dim_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_proj, embed_dim_, embed_dim_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(v_proj, embed_dim_, embed_dim_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(out_proj, embed_dim_, embed_dim_, true, dtype, device);
}

infinicore::Tensor SiglipAttention::forward(const infinicore::Tensor &hidden_states,
                                            const std::optional<infinicore::Tensor> &attention_mask) const {
    (void)attention_mask;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto q = q_proj_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    auto k = k_proj_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    auto v = v_proj_->forward(const_cast<infinicore::Tensor &>(hidden_states));

    auto q_reshaped = q->view({batch_size, seq_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous();
    auto k_reshaped = k->view({batch_size, seq_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous();
    auto v_reshaped = v->view({batch_size, seq_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous();

    auto q_flat = q_reshaped->view({batch_size * num_heads_, seq_len, head_dim_});
    auto k_flat = k_reshaped->view({batch_size * num_heads_, seq_len, head_dim_});
    auto v_flat = v_reshaped->view({batch_size * num_heads_, seq_len, head_dim_});

    auto k_t = k_flat->permute({0, 2, 1});
    auto attn_weights = infinilm::backends::ops::matmul(q_flat, k_t, scale_);

    auto attn_view = attn_weights->view({batch_size * num_heads_, seq_len, seq_len});
    infinilm::backends::ops::softmax_(attn_view, attn_view, -1);

    auto attn_output = infinilm::backends::ops::matmul(attn_weights, v_flat);
    auto out = attn_output->view({batch_size, num_heads_, seq_len, head_dim_})
                   ->permute({0, 2, 1, 3})
                   ->contiguous()
                   ->view({batch_size, seq_len, embed_dim_});

    return out_proj_->forward(out);
}

SiglipMLP::SiglipMLP(const nlohmann::json &config,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device)
    : activation_(config.value("hidden_act", "gelu")) {
    size_t hidden_size = config.value("hidden_size", 768);
    size_t intermediate_size = config.value("intermediate_size", 3072);
    INFINICORE_NN_MODULE_INIT(fc1, hidden_size, intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc2, intermediate_size, hidden_size, true, dtype, device);
}

infinicore::Tensor SiglipMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x = fc1_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    if (activation_ == "gelu_tanh") {
        x = infinilm::backends::ops::gelu_tanh(x);
    } else if (activation_ == "gelu") {
        x = infinilm::backends::ops::gelu(x);
    } else if (activation_ == "relu") {
        x = infinilm::backends::ops::relu(x);
    } else {
        throw std::runtime_error("SiglipMLP: unsupported activation " + activation_);
    }
    return fc2_->forward(x);
}

SiglipEncoderLayer::SiglipEncoderLayer(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    size_t hidden_size = config.value("hidden_size", 768);
    float layer_norm_eps = config.value("layer_norm_eps", 1e-6f);
    INFINICORE_NN_MODULE_INIT(layer_norm1, hidden_size, layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(layer_norm2, hidden_size, layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, dtype, device);
}

infinicore::Tensor SiglipEncoderLayer::forward(const infinicore::Tensor &hidden_states,
                                               const std::optional<infinicore::Tensor> &attention_mask) const {
    auto residual = hidden_states;
    auto x = layer_norm1_->forward(hidden_states);
    x = self_attn_->forward(x, attention_mask);
    x = infinilm::backends::ops::add(x, residual);

    residual = x;
    x = layer_norm2_->forward(x);
    x = mlp_->forward(x);
    x = infinilm::backends::ops::add(x, residual);
    return x;
}

SiglipEncoder::SiglipEncoder(const nlohmann::json &config,
                             const infinicore::DataType &dtype,
                             const infinicore::Device &device) {
    size_t num_layers = config.value("num_hidden_layers", 12);
    INFINICORE_NN_MODULE_VEC_INIT(layers, num_layers, SiglipEncoderLayer, config, dtype, device);
}

infinicore::Tensor SiglipEncoder::forward(const infinicore::Tensor &hidden_states,
                                          const std::optional<infinicore::Tensor> &attention_mask) const {
    auto x = hidden_states;
    for (const auto &layer : layers_) {
        x = layer->forward(x, attention_mask);
    }
    return x;
}

SiglipVisionModel::SiglipVisionModel(const nlohmann::json &config,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     bool drop_last_layer)
    : config_(config), drop_last_layer_(drop_last_layer) {
    INFINICORE_NN_MODULE_INIT(embeddings, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(encoder, config, dtype, device);
    size_t hidden_size = config.value("hidden_size", 768);
    float layer_norm_eps = config.value("layer_norm_eps", 1e-6f);
    INFINICORE_NN_MODULE_INIT(post_layernorm, hidden_size, layer_norm_eps, dtype, device);
}

infinicore::Tensor SiglipVisionModel::forward(const infinicore::Tensor &pixel_values,
                                              const std::optional<infinicore::Tensor> &tgt_sizes) const {
    auto hidden_states = embeddings_->forward(pixel_values, tgt_sizes);
    hidden_states = encoder_->forward(hidden_states, std::nullopt);
    return post_layernorm_->forward(hidden_states);
}

} // namespace infinilm::models::minicpmv
