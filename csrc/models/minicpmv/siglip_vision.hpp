#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <nlohmann/json.hpp>

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
    SiglipVisionEmbeddings(const nlohmann::json &config,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &tgt_sizes) const;

private:
    size_t hidden_size_;
    size_t patch_size_;
    size_t num_positions_;

    INFINICORE_NN_MODULE(SiglipPatchEmbedding, patch_embedding);
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, position_embedding);
};

class SiglipAttention : public infinicore::nn::Module {
public:
    SiglipAttention(const nlohmann::json &config,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const std::optional<infinicore::Tensor> &attention_mask) const;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;
    infinilm::backends::AttentionBackend attention_backend_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, out_proj);
};

class SiglipMLP : public infinicore::nn::Module {
public:
    SiglipMLP(const nlohmann::json &config,
              const infinicore::DataType &dtype,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    std::string activation_;
    INFINICORE_NN_MODULE(infinilm::nn::Linear, fc1);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, fc2);
};

class SiglipEncoderLayer : public infinicore::nn::Module {
public:
    SiglipEncoderLayer(const nlohmann::json &config,
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
    SiglipEncoder(const nlohmann::json &config,
                  const infinicore::DataType &dtype,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const std::optional<infinicore::Tensor> &attention_mask) const;

private:
    INFINICORE_NN_MODULE_VEC(SiglipEncoderLayer, layers);
};

class SiglipVisionModel : public infinicore::nn::Module {
public:
    SiglipVisionModel(const nlohmann::json &config,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device,
                      bool drop_last_layer);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &tgt_sizes) const;

private:
    nlohmann::json config_;
    bool drop_last_layer_{false};

    INFINICORE_NN_MODULE(SiglipVisionEmbeddings, embeddings);
    INFINICORE_NN_MODULE(SiglipEncoder, encoder);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, post_layernorm);
};

} // namespace infinilm::models::minicpmv
