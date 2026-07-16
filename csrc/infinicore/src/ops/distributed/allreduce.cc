#include "infinicore/ops/distributed/allreduce.hpp"
#include "../../utils.hpp"
#include "utils.hpp"

namespace infinicore::op::distributed {

struct PlannedMeta {
    graph::GraphTensor output, input;
    infinicclRedOp_t op;
    infinicclComm_t communicator;
};

AllReduce::AllReduce(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(output->numel() == input->numel());
    planned_meta_ = new PlannedMeta{graph::GraphTensor(output), graph::GraphTensor(input), op, communicator};
}
AllReduce::~AllReduce() {
    if (planned_meta_) {
        PlannedMeta *meta = reinterpret_cast<PlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void AllReduce::run() const {
    PlannedMeta *meta = reinterpret_cast<PlannedMeta *>(planned_meta_);

    detail::checkInfiniccl(
        "infinicclAllReduce",
        infinicclAllReduce(meta->input->data(),
                           meta->output->data(),
                           meta->input->numel(),
                           detail::toInfinicclDataType(meta->input->dtype()),
                           meta->op,
                           meta->communicator,
                           reinterpret_cast<void *>(infinicore::context::getStream())));
}

void AllReduce::execute(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(AllReduce, output, input, op, communicator);
}

Tensor allreduce(const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    allreduce_(output, input, op, communicator);
    return output;
}

void allreduce_(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT(output->dtype() == input->dtype());
    AllReduce::execute(output, input, op, communicator);
}
} // namespace infinicore::op::distributed
