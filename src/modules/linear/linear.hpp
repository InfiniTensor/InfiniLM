#pragma once

#include "../../dataloader/weights_loader.hpp"
#include "../../tensor.hpp"

namespace infinicore::nn::module {
class Linear {
protected:
    Linear() = default;

public:
    std::shared_ptr<Tensor> weight, bias;
    weights::DistributionType dist_type;
    int nranks = 1;
    static std::shared_ptr<Linear> init(
        size_t in_features,
        size_t out_features,
        infiniDtype_t dtype,
        bool has_bias = true,
        bool weight_transposed = false,
        weights::DistributionType dist_type = weights::DistributionType::FULL,
        int nranks = 1);
    void register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix = "", int rank = 0);
    void forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual = nullptr);
};
} // namespace infinicore::nn::module
