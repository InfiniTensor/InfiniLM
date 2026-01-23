#pragma once

#include "minicpmv_config.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/layernorm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <optional>
#include <vector>

namespace infinilm::models::minicpmv {

class SiglipPatchEmbedding : public infinicore::nn::Module {
public:
    SiglipPatchEmbedding(size_t in_channels,
                         size_t out_channels,
                         size_t patch_size,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

private:
    size_t patch_size_;
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);
};

class SiglipVisionEmbeddings : public infinicore::nn::Module {
public:
    SiglipVisionEmbeddings(const SiglipVisionConfig &config,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const std::optional<infinicore::Tensor> &tgt_sizes) const;

private:
    size_t hidden_size_;
    size_t patch_size_;
    size_t num_positions_;

    INFINICORE_NN_MODULE(SiglipPatchEmbedding, patch_embedding);
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, position_embedding);
};

class SiglipAttention : public infinicore::nn::Module {
public:
    SiglipAttention(const SiglipVisionConfig &config,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const std::optional<infinicore::Tensor> &attention_mask) const;

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

class SiglipMLP : public infinicore::nn::Module {
public:
    SiglipMLP(const SiglipVisionConfig &config,
              const infinicore::DataType &dtype,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    std::string activation_;
    INFINICORE_NN_MODULE(infinicore::nn::Linear, fc1);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, fc2);
};

class SiglipEncoderLayer : public infinicore::nn::Module {
public:
    SiglipEncoderLayer(const SiglipVisionConfig &config,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const std::optional<infinicore::Tensor> &attention_mask) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, layer_norm1);
    INFINICORE_NN_MODULE(SiglipAttention, self_attn);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, layer_norm2);
    INFINICORE_NN_MODULE(SiglipMLP, mlp);
};

class SiglipEncoder : public infinicore::nn::Module {
public:
    SiglipEncoder(const SiglipVisionConfig &config,
                  const infinicore::DataType &dtype,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const std::optional<infinicore::Tensor> &attention_mask) const;

private:
    INFINICORE_NN_MODULE_VEC(SiglipEncoderLayer, layers);
};

class SiglipVisionModel : public infinicore::nn::Module {
public:
    SiglipVisionModel(const SiglipVisionConfig &config,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device,
                      bool drop_last_layer);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const std::optional<infinicore::Tensor> &tgt_sizes) const;

private:
    SiglipVisionConfig config_;
    bool drop_last_layer_{false};

    INFINICORE_NN_MODULE(SiglipVisionEmbeddings, embeddings);
    INFINICORE_NN_MODULE(SiglipEncoder, encoder);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, post_layernorm);
};

} // namespace infinilm::models::minicpmv
