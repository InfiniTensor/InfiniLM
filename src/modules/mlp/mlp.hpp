#pragma once

#include "../../dataloader/weights_loader.hpp"
#include "../../tensor.hpp"

#include "../linear/linear.hpp"

namespace infinicore::nn::module {
class MLP {
private:
    MLP() = default;

public:
    std::shared_ptr<Linear> gate, up, down;

    static std::shared_ptr<MLP> init(size_t hidden_size, size_t intermediate_size, infiniDtype_t dtype, int nranks = 1);
    void register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix = "", int rank = 0);
    void forward(std::shared_ptr<Tensor> output, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> residual, std::shared_ptr<Tensor> gate_up_buf);
};
} // namespace infinicore::nn::module
