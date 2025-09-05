#pragma once

#include "../../dataloader/weights_loader.hpp"
#include "../../tensor.hpp"

namespace infinicore::nn::module {
class RMSNorm {
private:
    RMSNorm() = default;

public:
    std::shared_ptr<Tensor> weight;
    static std::shared_ptr<RMSNorm> init(size_t dim, infiniDtype_t dtype);
    void register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix = "", int rank = 0);
    void forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input);
};
} // namespace infinicore::nn::module
