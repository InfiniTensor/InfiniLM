#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Compressor : public infinicore::nn::Module {
public:
    DeepseekV4Compressor(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t compress_ratio,
                         size_t head_dim,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;
    std::vector<float> forward_values(const infinicore::Tensor &hidden_states,
                                      size_t &batch_size,
                                      size_t &num_blocks) const;
    size_t head_dim() const { return head_dim_; }
    size_t compress_ratio() const { return compress_ratio_; }
    size_t coff() const { return coff_; }

private:
    INFINICORE_NN_PARAMETER(ape);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wgate);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    size_t compress_ratio_{0};
    size_t head_dim_{0};
    size_t coff_{1};
};

} // namespace infinilm::models::deepseek_v4
