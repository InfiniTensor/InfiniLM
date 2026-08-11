#include "infinicore/ops/distributed/send_recv.hpp"
#include "../../utils.hpp"
#include "utils.hpp"

#include "infinicore/context/context.hpp"

#include <stdexcept>

namespace infinicore::op::distributed {
namespace {

void validateTensor(const Tensor &tensor) {
    INFINICORE_ASSERT(tensor);
    INFINICORE_ASSERT(tensor->is_contiguous());
    INFINICORE_ASSERT(tensor->numel() > 0);
    (void)detail::toInfinicclDataType(tensor->dtype());
}

void rejectGraphRecording() {
    if (infinicore::context::isGraphRecording()) {
        throw std::runtime_error(
            "InfiniCCL point-to-point communication cannot be recorded in a device graph");
    }
}

} // namespace

void send(const Tensor &input, int peer, infinicclComm_t communicator) {
    validateTensor(input);
    rejectGraphRecording();
    detail::checkInfiniccl(
        "infinicclSend",
        infinicclSend(input->data(),
                      input->numel(),
                      detail::toInfinicclDataType(input->dtype()),
                      peer,
                      communicator,
                      reinterpret_cast<void *>(infinicore::context::getStream())));
}

void recv_(Tensor output, int peer, infinicclComm_t communicator) {
    validateTensor(output);
    rejectGraphRecording();
    detail::checkInfiniccl(
        "infinicclRecv",
        infinicclRecv(output->data(),
                      output->numel(),
                      detail::toInfinicclDataType(output->dtype()),
                      peer,
                      communicator,
                      reinterpret_cast<void *>(infinicore::context::getStream())));
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
