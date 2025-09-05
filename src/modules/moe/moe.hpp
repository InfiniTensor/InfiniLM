#pragma once

#include "../../dataloader/weights_loader.hpp"
#include "../../tensor.hpp"

#include "../mlp/mlp.hpp"

namespace infinicore::nn::module {
class MoE {
private:
    MoE() = default;

public:
    std::shared_ptr<Linear> gate;
    std::vector<std::shared_ptr<MLP>> experts;
    size_t k; // top-k
    static std::shared_ptr<MoE> init(size_t n_experts, size_t n_experts_per_token, size_t hidden_size, size_t moe_intermediate_size, infiniDtype_t dtype);
    void register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix = "", int rank = 0, int nranks = 1);
    void forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual = nullptr);
};
} // namespace infinicore::nn::module