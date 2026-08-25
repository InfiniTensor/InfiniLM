#include "infinicore/ops/distributed/send_recv.hpp"
#include "../../utils.hpp"
#include "utils.hpp"

#include "infinicore/context/context.hpp"

namespace infinicore::op::distributed {
namespace {

void validateTensor(const Tensor &tensor) {
    INFINICORE_ASSERT(tensor);
    INFINICORE_ASSERT(tensor->is_contiguous());
    INFINICORE_ASSERT(tensor->numel() > 0);
    (void)detail::toInfinicclDataType(tensor->dtype());
}

struct SendPlannedMeta {
    graph::GraphTensor input;
    int peer;
    infinicclComm_t communicator;
};

struct RecvPlannedMeta {
    graph::GraphTensor output;
    int peer;
    infinicclComm_t communicator;
};

void runSend(const SendPlannedMeta &meta) {
    detail::checkInfiniccl(
        "infinicclSend",
        infinicclSend(meta.input->data(),
                      meta.input->numel(),
                      detail::toInfinicclDataType(meta.input->dtype()),
                      meta.peer,
                      meta.communicator,
                      reinterpret_cast<void *>(infinicore::context::getStream())));
}

void runRecv(const RecvPlannedMeta &meta) {
    detail::checkInfiniccl(
        "infinicclRecv",
        infinicclRecv(meta.output->data(),
                      meta.output->numel(),
                      detail::toInfinicclDataType(meta.output->dtype()),
                      meta.peer,
                      meta.communicator,
                      reinterpret_cast<void *>(infinicore::context::getStream())));
}

} // namespace

Send::Send(const Tensor &input, int peer, infinicclComm_t communicator) {
    validateTensor(input);
    planned_meta_ = new SendPlannedMeta{
        graph::GraphTensor(input), peer, communicator};
}

Send::~Send() {
    delete reinterpret_cast<SendPlannedMeta *>(planned_meta_);
    planned_meta_ = nullptr;
}

void Send::run() const {
    runSend(*reinterpret_cast<const SendPlannedMeta *>(planned_meta_));
}

void Send::execute(const Tensor &input,
                   int peer,
                   infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Send, input, peer, communicator);
}

Recv::Recv(Tensor output, int peer, infinicclComm_t communicator) {
    validateTensor(output);
    planned_meta_ = new RecvPlannedMeta{
        graph::GraphTensor(output), peer, communicator};
}

Recv::~Recv() {
    delete reinterpret_cast<RecvPlannedMeta *>(planned_meta_);
    planned_meta_ = nullptr;
}

void Recv::run() const {
    runRecv(*reinterpret_cast<const RecvPlannedMeta *>(planned_meta_));
}

void Recv::execute(Tensor output,
                   int peer,
                   infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Recv, output, peer, communicator);
}

void send(const Tensor &input, int peer, infinicclComm_t communicator) {
    Send::execute(input, peer, communicator);
}

void recv_(Tensor output, int peer, infinicclComm_t communicator) {
    Recv::execute(output, peer, communicator);
}

Tensor recv(const Shape &shape,
            DataType dtype,
            Device device,
            int peer,
            infinicclComm_t communicator) {
    auto output = Tensor::empty(shape, dtype, device);
    recv_(output, peer, communicator);
    return output;
}

} // namespace infinicore::op::distributed
