#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/normalization/layer_norm.hpp"
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"
#include <nlohmann/json.hpp>

#include <optional>
#include <vector>

namespace infinilm::models::minicpmv {

class SiglipPatchEmbedding : public infinilm::weights::ModelWeights {
public:
    SiglipPatchEmbedding(infinilm::runtime::Runtime &runtime, size_t in_channels,
                         size_t out_channels,
                         size_t patch_size,
                         const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values) const;

private:
    size_t patch_size_;
    infinilm::weights::Weight weight_;
    infinilm::weights::Weight bias_;
};

class SiglipVisionEmbeddings : public infinilm::weights::ModelWeights {
public:
    SiglipVisionEmbeddings(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                           const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                                      const infinilm::runtime::Tensor &tgt_sizes) const;

private:
    size_t hidden_size_;
    size_t patch_size_;
    size_t num_positions_;

    std::shared_ptr<SiglipPatchEmbedding> patch_embedding_;
    std::shared_ptr<infinilm::layers::Embedding> position_embedding_;
};

class SiglipAttention : public infinilm::weights::ModelWeights {
public:
    std::vector<std::shared_ptr<infinilm::weights::ModelWeights>> loading_children() const override {
        auto result = ModelWeights::loading_children();
        if (qkv_proj_) {
            result.push_back(qkv_proj_);
        }
        return result;
    }

    SiglipAttention(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                    const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                      const std::optional<infinilm::runtime::Tensor> &attention_mask,
                                      backends::AttentionBackend attention_backend) const;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> out_proj_;
};

class SiglipMLP : public infinilm::weights::ModelWeights {
public:
    SiglipMLP(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
              const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::string activation_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc2_;
};

class SiglipEncoderLayer : public infinilm::weights::ModelWeights {
public:
    SiglipEncoderLayer(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                       const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                      const std::optional<infinilm::runtime::Tensor> &attention_mask,
                                      backends::AttentionBackend attention_backend) const;

private:
    std::shared_ptr<infinilm::layers::LayerNorm> layer_norm1_;
    std::shared_ptr<SiglipAttention> self_attn_;
    std::shared_ptr<infinilm::layers::LayerNorm> layer_norm2_;
    std::shared_ptr<SiglipMLP> mlp_;
};

class SiglipEncoder : public infinilm::weights::ModelWeights {
public:
    SiglipEncoder(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                  const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                      const std::optional<infinilm::runtime::Tensor> &attention_mask,
                                      backends::AttentionBackend attention_backend) const;

private:
    std::vector<std::shared_ptr<SiglipEncoderLayer>> layers_;
};

class SiglipVisionModel : public infinilm::weights::ModelWeights {
public:
    SiglipVisionModel(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                      const infinilm::runtime::DataType &dtype,
                      bool drop_last_layer);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                                      const infinilm::runtime::Tensor &tgt_sizes,
                                      backends::AttentionBackend attention_backend) const;

private:
    nlohmann::json config_;
    bool drop_last_layer_{false};

    std::shared_ptr<SiglipVisionEmbeddings> embeddings_;
    std::shared_ptr<SiglipEncoder> encoder_;
    std::shared_ptr<infinilm::layers::LayerNorm> post_layernorm_;
};

} // namespace infinilm::models::minicpmv
