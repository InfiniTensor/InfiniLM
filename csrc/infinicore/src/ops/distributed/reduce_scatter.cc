#include "infinicore/ops/distributed/reduce_scatter.hpp"
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

void validate_scatter_input(const Tensor &input,
                            const std::vector<size_t> &split_sizes) {
    INFINICORE_ASSERT(input);
    INFINICORE_ASSERT(input->is_contiguous());
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(input->shape()[0] > 0);
    INFINICORE_ASSERT(!split_sizes.empty());
    (void)detail::toInfinicclDataType(input->dtype());
    const size_t local_dim0 = split_sizes.front();
    if (!all_equal_to(split_sizes, local_dim0)) {
        throw std::runtime_error("InfiniCCL does not support variable-count reduce-scatter");
    }
    INFINICORE_ASSERT(local_dim0 > 0);
    INFINICORE_ASSERT(local_dim0 <= std::numeric_limits<size_t>::max() / split_sizes.size());
    INFINICORE_ASSERT(input->shape()[0] == local_dim0 * split_sizes.size());
}

void validate_reduce_scatterv(Tensor output,
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
    if (!all_equal_to(split_sizes, output->shape()[0])) {
        throw std::runtime_error("InfiniCCL does not support variable-count reduce-scatter");
    }
    INFINICORE_ASSERT(output->shape()[0] > 0);
    INFINICORE_ASSERT(output->shape()[0] <= std::numeric_limits<size_t>::max() / split_sizes.size());
    INFINICORE_ASSERT(input->shape()[0] == output->shape()[0] * split_sizes.size());
    for (size_t dim = 1; dim < input->ndim(); ++dim) {
        INFINICORE_ASSERT(output->shape()[dim] == input->shape()[dim]);
    }
    INFINICORE_ASSERT(output->numel() <= std::numeric_limits<size_t>::max() / split_sizes.size());
    INFINICORE_ASSERT(input->numel() == output->numel() * split_sizes.size());
}

} // namespace

struct ReduceScatterPlannedMeta {
    graph::GraphTensor output, input;
    infinicclRedOp_t op;
    infinicclComm_t communicator;
};

ReduceScatter::ReduceScatter(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(input->numel() > 0);
    INFINICORE_ASSERT(output->numel() > 0);
    INFINICORE_ASSERT(input->numel() % output->numel() == 0);
    planned_meta_ = new ReduceScatterPlannedMeta{graph::GraphTensor(output), graph::GraphTensor(input), op, communicator};
}

ReduceScatter::~ReduceScatter() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<ReduceScatterPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void ReduceScatter::run() const {
    auto *meta = reinterpret_cast<ReduceScatterPlannedMeta *>(planned_meta_);
    detail::checkInfiniccl(
        "infinicclReduceScatter",
        infinicclReduceScatter(meta->input->data(),
                               meta->output->data(),
                               meta->output->numel(),
                               detail::toInfinicclDataType(meta->input->dtype()),
                               meta->op,
                               meta->communicator,
                               reinterpret_cast<void *>(infinicore::context::getStream())));
}

void ReduceScatter::execute(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    INFINICORE_ASSERT(input->numel() > 0);
    INFINICORE_ASSERT(output->numel() > 0);
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(ReduceScatter, output, input, op, communicator);
}

Tensor reduce_scatter(const Tensor &input, size_t world_size, infinicclRedOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(world_size > 0);
    INFINICORE_ASSERT(input->shape()[0] % world_size == 0);
    auto shape = input->shape();
    shape[0] /= world_size;
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    reduce_scatter_(output, input, op, communicator);
    return output;
}

void reduce_scatter_(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    ReduceScatter::execute(output, input, op, communicator);
}

Tensor reduce_scatterv(const Tensor &input,
                       const std::vector<size_t> &split_sizes,
                       size_t rank,
                       infinicclRedOp_t op,
                       infinicclComm_t communicator) {
    INFINICORE_ASSERT(!split_sizes.empty());
    validate_scatter_input(input, split_sizes);
    INFINICORE_ASSERT(rank < split_sizes.size());
    auto shape = input->shape();
    shape[0] = split_sizes[rank];
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    reduce_scatterv_(output, input, split_sizes, op, communicator);
    return output;
}

void reduce_scatterv_(Tensor output,
                      const Tensor &input,
                      const std::vector<size_t> &split_sizes,
                      infinicclRedOp_t op,
                      infinicclComm_t communicator) {
    validate_reduce_scatterv(output, input, split_sizes);
    ReduceScatter::execute(output, input, op, communicator);
}

std::vector<Tensor> reduce_scatterv_many(const std::vector<Tensor> &inputs,
                                         const std::vector<size_t> &split_sizes,
                                         size_t rank,
                                         infinicclRedOp_t op,
                                         infinicclComm_t communicator) {
    INFINICORE_ASSERT(!split_sizes.empty());
    INFINICORE_ASSERT(rank < split_sizes.size());
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    for (const auto &input : inputs) {
        validate_scatter_input(input, split_sizes);
    }
    for (const auto &input : inputs) {
        auto shape = input->shape();
        shape[0] = split_sizes[rank];
        outputs.push_back(Tensor::empty(shape, input->dtype(), input->device()));
    }
    reduce_scatterv_many_(outputs, inputs, split_sizes, op, communicator);
    return outputs;
}

void reduce_scatterv_many_(const std::vector<Tensor> &outputs,
                           const std::vector<Tensor> &inputs,
                           const std::vector<size_t> &split_sizes,
                           infinicclRedOp_t op,
                           infinicclComm_t communicator) {
    INFINICORE_ASSERT(outputs.size() == inputs.size());
    INFINICORE_ASSERT(!split_sizes.empty());
    if (inputs.empty()) {
        return;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        validate_reduce_scatterv(outputs[i], inputs[i], split_sizes);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        ReduceScatter::execute(outputs[i], inputs[i], op, communicator);
    }
}

} // namespace infinicore::op::distributed
