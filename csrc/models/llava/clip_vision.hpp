#pragma once

#include "llava_config.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/layernorm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace infinilm::models::llava {

class ClipPatchEmbedding : public infinicore::nn::Module {
public:
    ClipPatchEmbedding(size_t in_channels,
                       size_t out_channels,
                       size_t patch_size,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

private:
    size_t patch_size_;
    INFINICORE_NN_PARAMETER(weight);
    infinicore::Tensor bias_;
};

class ClipVisionEmbeddings : public infinicore::nn::Module {
public:
    ClipVisionEmbeddings(const ClipVisionConfig &config,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

    size_t num_patches() const { return num_patches_; }
    size_t num_positions() const { return num_positions_; }

    infinicore::Tensor class_embedding() const { return class_embedding_; }
    infinicore::Tensor position_embedding_weight() const { return position_embedding_->weight(); }

private:
    size_t hidden_size_;
    size_t patch_size_;
    size_t num_patches_;
    size_t num_positions_;

    INFINICORE_NN_PARAMETER(class_embedding);
    INFINICORE_NN_MODULE(ClipPatchEmbedding, patch_embedding);
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, position_embedding);
};

class ClipAttention : public infinicore::nn::Module {
public:
    ClipAttention(const ClipVisionConfig &config,
                  const infinicore::DataType &dtype,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    INFINICORE_NN_MODULE(infinicore::nn::Linear, q_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, k_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, v_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, out_proj);
};

class ClipMLP : public infinicore::nn::Module {
public:
    ClipMLP(const ClipVisionConfig &config,
            const infinicore::DataType &dtype,
            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, fc1);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, fc2);
};

class ClipEncoderLayer : public infinicore::nn::Module {
public:
    ClipEncoderLayer(const ClipVisionConfig &config,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, layer_norm1);
    INFINICORE_NN_MODULE(ClipAttention, self_attn);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, layer_norm2);
    INFINICORE_NN_MODULE(ClipMLP, mlp);
};

class ClipEncoder : public infinicore::nn::Module {
public:
    ClipEncoder(const ClipVisionConfig &config,
                const infinicore::DataType &dtype,
                const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    size_t num_layers() const { return layers_.size(); }
    const std::vector<std::shared_ptr<ClipEncoderLayer>> &layers() const { return layers_; }

private:
    INFINICORE_NN_MODULE_VEC(ClipEncoderLayer, layers);
};

class ClipVisionModel : public infinicore::nn::Module {
public:
    ClipVisionModel(const ClipVisionConfig &config,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device);

    // Return the selected hidden state for LLaVA (matching HF hidden_states indexing).
    infinicore::Tensor forward_features(const infinicore::Tensor &pixel_values,
                                        int64_t feature_layer) const;

private:
    ClipVisionConfig config_;

    INFINICORE_NN_MODULE(ClipVisionEmbeddings, embeddings);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, pre_layrnorm);
    INFINICORE_NN_MODULE(ClipEncoder, encoder);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, post_layernorm);
};

} // namespace infinilm::models::llava
