#pragma once
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <infiniccl.h>
namespace infinilm::models::glm_moe_dsa {
class GlmVocabEmbedding final : public infinicore::nn::Module {
public:
    GlmVocabEmbedding(size_t, size_t, const infinicore::DataType &, const infinicore::Device &);
    infinicore::Tensor forward(const infinicore::Tensor &) const;

private:
    INFINICORE_NN_PARAMETER(weight);
    size_t hidden_{0}, start_{0}, end_{0};
    infinicclComm_t comm_{nullptr};
};
class GlmVocabLMHead final : public infinicore::nn::Module {
public:
    GlmVocabLMHead(size_t, size_t, bool, const infinicore::DataType &, const infinicore::Device &);
    infinicore::Tensor forward(const infinicore::Tensor &) const;

private:
    INFINICORE_NN_PARAMETER(weight);
    size_t vocab_{0}, world_{1};
    infinicclComm_t comm_{nullptr};
};
} // namespace infinilm::models::glm_moe_dsa
