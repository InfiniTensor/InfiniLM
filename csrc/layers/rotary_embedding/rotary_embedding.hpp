#pragma once

#include "../../config/model_config.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::layers::rotary_embedding {

class RotaryEmbedding {
public:
    RotaryEmbedding(size_t head_dim,
                    size_t max_seq_len,
                    double theta,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device,
                    std::shared_ptr<infinicore::nn::RoPE::ScalingConfig> scaling);

    void forward_pair(const infinicore::Tensor &query_out,
                      const infinicore::Tensor &query,
                      const infinicore::Tensor &key_out,
                      const infinicore::Tensor &key,
                      const infinicore::Tensor &positions) const;

    void forward_pair_inplace(infinicore::Tensor &query,
                              infinicore::Tensor &key,
                              const infinicore::Tensor &positions) const;

private:
    void initialize_cos_sin_cache();

    bool try_infiniops(const infinicore::Tensor &query_out,
                       const infinicore::Tensor &query,
                       const infinicore::Tensor &key_out,
                       const infinicore::Tensor &key,
                       const infinicore::Tensor &positions) const;

    size_t head_dim_;
    size_t max_seq_len_;
    double theta_;
    infinicore::DataType dtype_;
    infinicore::Device device_;
    std::shared_ptr<infinicore::nn::RoPE::ScalingConfig> scaling_;
    std::shared_ptr<infinicore::nn::RoPE> legacy_;
    infinicore::Tensor cos_sin_cache_;
};

std::shared_ptr<RotaryEmbedding> get_rotary_embedding(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                                      const infinicore::Device &device);

std::shared_ptr<infinicore::nn::RoPE> get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                               const infinicore::Device &device);

} // namespace infinilm::layers::rotary_embedding
