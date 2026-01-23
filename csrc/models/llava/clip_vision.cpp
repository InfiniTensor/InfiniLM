#include "clip_vision.hpp"

#include "infinicore/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace infinilm::models::llava {

ClipPatchEmbedding::ClipPatchEmbedding(size_t in_channels,
                                       size_t out_channels,
                                       size_t patch_size,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : patch_size_(patch_size) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_channels, in_channels, patch_size_, patch_size_}, dtype, device));
    bias_ = infinicore::Tensor::zeros({out_channels}, dtype, device);
}

infinicore::Tensor ClipPatchEmbedding::forward(const infinicore::Tensor &pixel_values) const {
    auto shape = pixel_values->shape();
    if (shape.size() != 4) {
        throw std::runtime_error("ClipPatchEmbedding: expected 4D pixel_values");
    }
    size_t batch_size = shape[0];
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    if (height % patch_size_ != 0 || width % patch_size_ != 0) {
        throw std::runtime_error("ClipPatchEmbedding: image size must be divisible by patch size");
    }

    size_t h_patches = height / patch_size_;
    size_t w_patches = width / patch_size_;
    auto patches = pixel_values->view({batch_size, channels, h_patches, patch_size_, w_patches, patch_size_})
                       ->permute({0, 2, 4, 1, 3, 5})
                       ->contiguous();
    auto patches2d = patches->view({batch_size * h_patches * w_patches, channels * patch_size_ * patch_size_});
    auto weight2d = weight_->view({weight_->size(0), channels * patch_size_ * patch_size_});
    auto out2d = infinicore::op::linear(
        patches2d, weight2d, std::make_optional<infinicore::Tensor>(bias_));
    auto out = out2d->view({batch_size, h_patches, w_patches, weight_->size(0)})->permute({0, 3, 1, 2});
    return out->contiguous();
}

ClipVisionEmbeddings::ClipVisionEmbeddings(const ClipVisionConfig &config,
                                           const infinicore::DataType &dtype,
                                           const infinicore::Device &device)
    : hidden_size_(config.hidden_size),
      patch_size_(config.patch_size),
      num_patches_((config.image_size / config.patch_size) * (config.image_size / config.patch_size)),
      num_positions_(num_patches_ + 1) {
    INFINICORE_NN_PARAMETER_INIT(class_embedding, ({hidden_size_}, dtype, device));
    INFINICORE_NN_MODULE_INIT(patch_embedding, 3, hidden_size_, patch_size_, dtype, device);
    INFINICORE_NN_MODULE_INIT(position_embedding, num_positions_, hidden_size_, std::nullopt, dtype, device);
}

infinicore::Tensor ClipVisionEmbeddings::forward(const infinicore::Tensor &pixel_values) const {
    if (!pixel_values) {
        throw std::runtime_error("ClipVisionEmbeddings: pixel_values is empty");
    }

    auto batch_size = pixel_values->size(0);

    // Patch embedding: [B, 3, H, W] -> [B, hidden, H/patch, W/patch]
    auto patch_embeds = patch_embedding_->forward(pixel_values);

    auto grid_h = patch_embeds->size(2);
    auto grid_w = patch_embeds->size(3);
    auto seq_len = grid_h * grid_w;

    // Flatten to [B, seq_len, hidden]
    auto patch_flat = patch_embeds->view({batch_size, hidden_size_, seq_len})->permute({0, 2, 1});

    // Build class token [B, 1, hidden]
    auto class_token = infinicore::Tensor::empty({batch_size, 1, hidden_size_}, patch_flat->dtype(), patch_flat->device());
    auto class_src = class_embedding_->view({1, hidden_size_});
    if (class_src->shape().size() == 2) {
        class_src = class_src->unsqueeze(0);
    }
    for (size_t b = 0; b < batch_size; ++b) {
        auto dst = class_token->narrow({{0, b, 1}});
        dst->copy_from(class_src);
    }

    // Concat class token and patch embeddings
    auto embeddings = infinicore::Tensor::empty({batch_size, seq_len + 1, hidden_size_}, patch_flat->dtype(), patch_flat->device());
    embeddings->narrow({{1, 0, 1}})->copy_from(class_token);
    embeddings->narrow({{1, 1, seq_len}})->copy_from(patch_flat);

    // Add position embedding
    // Position ids: [B, seq_len + 1]
    auto pos_ids_cpu = infinicore::Tensor::empty({batch_size, seq_len + 1}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *pos_ptr = reinterpret_cast<int64_t *>(pos_ids_cpu->data());
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < seq_len + 1; ++i) {
            pos_ptr[b * (seq_len + 1) + i] = static_cast<int64_t>(i);
        }
    }
    auto pos_ids = pos_ids_cpu->to(embeddings->device());
    auto pos_embed = position_embedding_->forward(pos_ids);
    return infinicore::op::add(embeddings, pos_embed);
}

ClipAttention::ClipAttention(const ClipVisionConfig &config,
                             const infinicore::DataType &dtype,
                             const infinicore::Device &device)
    : embed_dim_(config.hidden_size),
      num_heads_(config.num_attention_heads),
      head_dim_(config.hidden_size / config.num_attention_heads),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (embed_dim_ % num_heads_ != 0) {
        throw std::runtime_error("ClipAttention: embed_dim must be divisible by num_heads");
    }
    INFINICORE_NN_MODULE_INIT(q_proj, embed_dim_, embed_dim_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_proj, embed_dim_, embed_dim_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(v_proj, embed_dim_, embed_dim_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(out_proj, embed_dim_, embed_dim_, true, dtype, device);
}

infinicore::Tensor ClipAttention::forward(const infinicore::Tensor &hidden_states) const {
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
    auto attn_weights = infinicore::op::matmul(q_flat, k_t, scale_);
    auto attn_view = attn_weights->view({batch_size * num_heads_, seq_len, seq_len});
    infinicore::op::softmax_(attn_view, attn_view, -1);

    auto attn_output = infinicore::op::matmul(attn_weights, v_flat);
    auto out = attn_output->view({batch_size, num_heads_, seq_len, head_dim_})
                   ->permute({0, 2, 1, 3})
                   ->contiguous()
                   ->view({batch_size, seq_len, embed_dim_});

    return out_proj_->forward(out);
}

ClipMLP::ClipMLP(const ClipVisionConfig &config,
                 const infinicore::DataType &dtype,
                 const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(fc1, config.hidden_size, config.intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc2, config.intermediate_size, config.hidden_size, true, dtype, device);
}

infinicore::Tensor ClipMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x = fc1_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    auto x_act = infinicore::op::quick_gelu(x);
    return fc2_->forward(x_act);
}

ClipEncoderLayer::ClipEncoderLayer(const ClipVisionConfig &config,
                                   const infinicore::DataType &dtype,
                                   const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(layer_norm1, config.hidden_size, config.layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(layer_norm2, config.hidden_size, config.layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, dtype, device);
}

infinicore::Tensor ClipEncoderLayer::forward(const infinicore::Tensor &hidden_states) const {
    auto residual = hidden_states;
    auto x = layer_norm1_->forward(hidden_states);
    x = self_attn_->forward(x);
    x = infinicore::op::add(x, residual);

    residual = x;
    x = layer_norm2_->forward(x);
    x = mlp_->forward(x);
    x = infinicore::op::add(x, residual);
    return x;
}

ClipEncoder::ClipEncoder(const ClipVisionConfig &config,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device) {
    INFINICORE_NN_MODULE_VEC_INIT(layers, config.num_hidden_layers, ClipEncoderLayer, config, dtype, device);
}

infinicore::Tensor ClipEncoder::forward(const infinicore::Tensor &hidden_states) const {
    auto x = hidden_states;
    for (const auto &layer : layers_) {
        x = layer->forward(x);
    }
    return x;
}

ClipVisionModel::ClipVisionModel(const ClipVisionConfig &config,
                                 const infinicore::DataType &dtype,
                                 const infinicore::Device &device)
    : config_(config) {
    INFINICORE_NN_MODULE_INIT(embeddings, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(pre_layrnorm, config.hidden_size, config.layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(encoder, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_layernorm, config.hidden_size, config.layer_norm_eps, dtype, device);
}

infinicore::Tensor ClipVisionModel::forward_features(const infinicore::Tensor &pixel_values,
                                                     int64_t feature_layer) const {
    auto hidden_states = embeddings_->forward(pixel_values);
    hidden_states = pre_layrnorm_->forward(hidden_states);

    const auto total_hidden_states = static_cast<int64_t>(config_.num_hidden_layers) + 1;
    int64_t target = feature_layer >= 0 ? feature_layer : (total_hidden_states + feature_layer);
    if (target < 0 || target >= total_hidden_states) {
        throw std::runtime_error("ClipVisionModel: invalid feature_layer");
    }

    int64_t hidden_idx = 0;
    if (hidden_idx == target) {
        return hidden_states;
    }

    const auto &layers = encoder_->layers();
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        hidden_states = layers[layer_idx]->forward(hidden_states);
        hidden_idx++;
        if (hidden_idx == target) {
            return hidden_states;
        }
    }

    return hidden_states;
}

} // namespace infinilm::models::llava
