#pragma once

#include "../../tensor.hpp"

#include <infiniccl/infiniccl.h>

namespace infinicore::op::distributed {

void send(const Tensor &input, int peer, infinicclComm_t communicator);
void recv_(Tensor output, int peer, infinicclComm_t communicator);
Tensor recv(const Shape &shape,
            DataType dtype,
            Device device,
            int peer,
            infinicclComm_t communicator);

} // namespace infinicore::op::distributed
