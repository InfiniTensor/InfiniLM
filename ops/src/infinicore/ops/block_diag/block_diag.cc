#include "infinicore/ops/block_diag.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<BlockDiag::schema> &BlockDiag::dispatcher() {
    static common::OpDispatcher<schema> dispatcher_;
    return dispatcher_;
}

void BlockDiag::execute(Tensor output, const std::vector<Tensor> &inputs) {
    if (inputs.empty()) {
        throw std::runtime_error("block_diag expects at least one input tensor");
    }

    for (const auto &x : inputs) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, x);
    }

    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No BlockDiag implementation found for device type: "
            + std::to_string(static_cast<int>(device_type)));
    }

    func(output, inputs);
}

Tensor block_diag(const std::vector<Tensor> &inputs) {
    if (inputs.empty()) {
        throw std::runtime_error("block_diag expects at least one input tensor");
    }

    const auto &device = inputs.front()->device();
    const auto dtype = inputs.front()->dtype();

    Size total_rows = 0;
    Size total_cols = 0;
    for (const auto &x : inputs) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(inputs.front(), x);
        INFINICORE_ASSERT(x->dtype() == dtype);
        INFINICORE_ASSERT(x->ndim() == 2);
        total_rows += x->size(0);
        total_cols += x->size(1);
    }

    auto output = Tensor::empty({total_rows, total_cols}, dtype, device);
    block_diag_(output, inputs);
    return output;
}

void block_diag_(Tensor output, const std::vector<Tensor> &inputs) {
    BlockDiag::execute(output, inputs);
}

} // namespace infinicore::op
