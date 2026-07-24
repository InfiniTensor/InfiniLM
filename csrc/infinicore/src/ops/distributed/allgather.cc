#include "infinicore/ops/distributed/allgather.hpp"
#include "../../utils.hpp"
#include "utils.hpp"

#include "infinicore/context/context.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infinicore::op::distributed {
namespace {

bool all_equal_to(const std::vector<size_t> &values, size_t expected) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [&](size_t value) {
        return value == expected;
    });
}

void validate_gather_input(const Tensor &input, const std::vector<size_t> &split_sizes) {
    INFINICORE_ASSERT(input);
    INFINICORE_ASSERT(input->is_contiguous());
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(input->shape()[0] > 0);
    INFINICORE_ASSERT(!split_sizes.empty());
    (void)detail::toInfinicclDataType(input->dtype());
    if (!all_equal_to(split_sizes, input->shape()[0])) {
        throw std::runtime_error("InfiniCCL does not support variable-count all-gather");
    }
    INFINICORE_ASSERT(input->shape()[0] <= std::numeric_limits<size_t>::max() / split_sizes.size());
}

void validate_allgatherv(Tensor output,
                         const Tensor &input,
                         const std::vector<size_t> &split_sizes) {
    INFINICORE_ASSERT(output && input);
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    (void)detail::toInfinicclDataType(input->dtype());
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(output->ndim() == input->ndim());
    INFINICORE_ASSERT(input->shape()[0] > 0);
    INFINICORE_ASSERT(!split_sizes.empty());
    if (!all_equal_to(split_sizes, input->shape()[0])) {
        throw std::runtime_error("InfiniCCL does not support variable-count all-gather");
    }
    INFINICORE_ASSERT(input->shape()[0] <= std::numeric_limits<size_t>::max() / split_sizes.size());
    INFINICORE_ASSERT(output->shape()[0] == input->shape()[0] * split_sizes.size());
    for (size_t dim = 1; dim < input->ndim(); ++dim) {
        INFINICORE_ASSERT(output->shape()[dim] == input->shape()[dim]);
    }
    INFINICORE_ASSERT(input->numel() <= std::numeric_limits<size_t>::max() / split_sizes.size());
    INFINICORE_ASSERT(output->numel() == input->numel() * split_sizes.size());
}

} // namespace

struct AllGatherPlannedMeta {
    graph::GraphTensor output, input;
    infinicclComm_t communicator;
};

AllGather::AllGather(Tensor output, const Tensor &input, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(input->numel() > 0);
    INFINICORE_ASSERT(output->numel() > 0);
    INFINICORE_ASSERT(output->numel() % input->numel() == 0);
    planned_meta_ = new AllGatherPlannedMeta{graph::GraphTensor(output), graph::GraphTensor(input), communicator};
}

AllGather::~AllGather() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<AllGatherPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void AllGather::run() const {
    auto *meta = reinterpret_cast<AllGatherPlannedMeta *>(planned_meta_);
    detail::checkInfiniccl(
        "infinicclAllGather",
        infinicclAllGather(meta->input->data(),
                           meta->output->data(),
                           meta->input->numel(),
                           detail::toInfinicclDataType(meta->input->dtype()),
                           meta->communicator,
                           reinterpret_cast<void *>(infinicore::context::getStream())));
}

void AllGather::execute(Tensor output, const Tensor &input, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    INFINICORE_ASSERT(input->numel() > 0);
    INFINICORE_ASSERT(output->numel() > 0);
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(AllGather, output, input, communicator);
}

Tensor allgather(const Tensor &input, size_t world_size, infinicclComm_t communicator) {
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(world_size > 0);
    auto shape = input->shape();
    shape[0] *= world_size;
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    allgather_(output, input, communicator);
    return output;
}

void allgather_(Tensor output, const Tensor &input, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    AllGather::execute(output, input, communicator);
}

Tensor allgatherv(const Tensor &input, const std::vector<size_t> &split_sizes, infinicclComm_t communicator) {
    validate_gather_input(input, split_sizes);
    auto shape = input->shape();
    shape[0] *= split_sizes.size();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    allgatherv_(output, input, split_sizes, communicator);
    return output;
}

void allgatherv_(Tensor output, const Tensor &input, const std::vector<size_t> &split_sizes, infinicclComm_t communicator) {
    validate_allgatherv(output, input, split_sizes);
    AllGather::execute(output, input, communicator);
}

std::vector<Tensor> allgatherv_many(const std::vector<Tensor> &inputs,
                                    const std::vector<size_t> &split_sizes,
                                    infinicclComm_t communicator) {
    INFINICORE_ASSERT(!split_sizes.empty());
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    if (inputs.empty()) {
        return outputs;
    }
    for (const auto &input : inputs) {
        validate_gather_input(input, split_sizes);
    }
    const size_t total_dim0 = inputs.front()->shape()[0] * split_sizes.size();
    for (const auto &input : inputs) {
        auto shape = input->shape();
        shape[0] = total_dim0;
        outputs.push_back(Tensor::empty(shape, input->dtype(), input->device()));
    }
    allgatherv_many_(outputs, inputs, split_sizes, communicator);
    return outputs;
}

void allgatherv_many_(const std::vector<Tensor> &outputs,
                      const std::vector<Tensor> &inputs,
                      const std::vector<size_t> &split_sizes,
                      infinicclComm_t communicator) {
    INFINICORE_ASSERT(outputs.size() == inputs.size());
    INFINICORE_ASSERT(!split_sizes.empty());
    if (inputs.empty()) {
        return;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        validate_allgatherv(outputs[i], inputs[i], split_sizes);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        AllGather::execute(outputs[i], inputs[i], communicator);
    }
}

} // namespace infinicore::op::distributed
