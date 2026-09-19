#pragma once

#include "infinicore/nn/rmsnorm.hpp"
#include "qwen3_5_decoderLayer.hpp"

namespace infinilm::models::qwen3_5 {

// The checkpoint's one-step MTP head shares the target embedding and LM head.
class Qwen35MTP : public infinicore::nn::Module {
public:
    Qwen35MTP(std::shared_ptr<config::ModelConfig> config, const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &embeddings,
                               const infinicore::Tensor &target_hidden,
                               const infinicore::Tensor &positions) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, pre_fc_norm_embedding);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, pre_fc_norm_hidden);
    INFINICORE_NN_MODULE(layers::linear::ReplicatedLinear, fc);
    INFINICORE_NN_MODULE(Qwen35DecoderLayer, layer);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};

} // namespace infinilm::models::qwen3_5
