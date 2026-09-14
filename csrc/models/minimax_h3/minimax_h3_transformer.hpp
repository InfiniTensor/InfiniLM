#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/nn/rope.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace infinilm::models::minimax_h3 {

class MiniMaxH3Attention : public infinicore::nn::Module {
public:
    MiniMaxH3Attention(size_t hidden_size,
                       size_t num_heads,
                       size_t head_dim,
                       double qk_norm_eps,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(
        const infinicore::Tensor &hidden_states,
        const std::shared_ptr<infinicore::nn::RoPE> &rope = nullptr,
        const std::optional<infinicore::Tensor> &position_ids = std::nullopt,
        const std::optional<infinicore::Tensor> &rotary_cos_sin_cache = std::nullopt) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    size_t num_heads_;
    size_t head_dim_;
    size_t inner_dim_;
    float scale_;

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm_q);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm_k);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, out_proj);
};

class MiniMaxH3MLP : public infinicore::nn::Module {
public:
    MiniMaxH3MLP(size_t hidden_size,
                 size_t ffn_hidden_size,
                 const infinicore::DataType &dtype,
                 const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, down_proj);
};

class MiniMaxH3TokenRefinerBlock : public infinicore::nn::Module {
public:
    MiniMaxH3TokenRefinerBlock(size_t hidden_size,
                               size_t num_heads,
                               size_t head_dim,
                               size_t ffn_hidden_size,
                               double norm_eps,
                               double qk_norm_eps,
                               const infinicore::DataType &dtype,
                               const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm1);
    INFINICORE_NN_MODULE(MiniMaxH3Attention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm2);
    INFINICORE_NN_MODULE(MiniMaxH3MLP, ff);
};

class MiniMaxH3TokenRefiner : public infinicore::nn::Module {
public:
    MiniMaxH3TokenRefiner(size_t hidden_size,
                          size_t num_heads,
                          size_t head_dim,
                          size_t ffn_hidden_size,
                          size_t num_layers,
                          double norm_eps,
                          double qk_norm_eps,
                          double final_norm_eps,
                          const infinicore::DataType &dtype,
                          const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE_VEC(MiniMaxH3TokenRefinerBlock, refiner_blocks);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, final_norm);
};

class MiniMaxH3AdaLNProjection : public infinicore::nn::Module {
public:
    MiniMaxH3AdaLNProjection(size_t in_features,
                             size_t out_features,
                             const infinicore::DataType &dtype,
                             const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &temb) const;

private:
    infinicore::DataType dtype_;
    size_t tp_size_;
    infinicclComm_t communicator_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, linear);
};

class MiniMaxH3TransformerBlock : public infinicore::nn::Module {
public:
    MiniMaxH3TransformerBlock(size_t hidden_size,
                              size_t num_heads,
                              size_t head_dim,
                              size_t ffn_hidden_size,
                              size_t time_embed_dim,
                              double norm_eps,
                              double qk_norm_eps,
                              const infinicore::DataType &dtype,
                              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &temb,
                               const infinicore::Tensor &adaln_indices,
                               const std::shared_ptr<infinicore::nn::RoPE> &rope,
                               const infinicore::Tensor &position_ids,
                               const std::optional<infinicore::Tensor> &rotary_cos_sin_cache = std::nullopt) const;

private:
    size_t hidden_size_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm1);
    INFINICORE_NN_MODULE(MiniMaxH3Attention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm2);
    INFINICORE_NN_MODULE(MiniMaxH3MLP, ff);
    INFINICORE_NN_MODULE(MiniMaxH3AdaLNProjection, adaln_proj);
};

class MiniMaxH3TimeEmbedder : public infinicore::nn::Module {
public:
    MiniMaxH3TimeEmbedder(size_t input_dim,
                          size_t hidden_size,
                          size_t output_dim,
                          const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &timestep) const;

private:
    size_t input_dim_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_2);
};

class MiniMaxH3OutputNorm : public infinicore::nn::Module {
public:
    MiniMaxH3OutputNorm(size_t hidden_size,
                        size_t time_embed_dim,
                        double norm_eps,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &temb,
                               const infinicore::Tensor &timestep_indices) const;

private:
    size_t hidden_size_;
    infinicore::DataType dtype_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear);
};

class MiniMaxH3Transformer : public infinilm::InfinilmModel {
public:
    MiniMaxH3Transformer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    size_t hidden_size_;
    infinicore::DataType dtype_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj_in);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, audio_proj_in);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, context_embedder);
    INFINICORE_NN_MODULE(MiniMaxH3TimeEmbedder, time_embedder);
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rope);
    INFINICORE_NN_MODULE(MiniMaxH3TokenRefiner, token_refiner);
    INFINICORE_NN_MODULE_VEC(MiniMaxH3TransformerBlock, transformer_blocks);
    INFINICORE_NN_MODULE(MiniMaxH3OutputNorm, norm_out);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj_out);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, audio_proj_out);
};

std::shared_ptr<infinilm::config::ModelConfig>
create_minimax_h3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minimax_h3
