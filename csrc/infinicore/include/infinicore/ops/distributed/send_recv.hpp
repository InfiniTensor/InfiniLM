#pragma once

#include "../../graph/graph.hpp"

#include <infiniccl/infiniccl.h>

namespace infinicore::op::distributed {

class Send : public graph::GraphOperator {
public:
    Send(const Tensor &input, int peer, infinicclComm_t communicator);
    ~Send() override;

    void run() const override;
    bool is_device_graph_capture_safe() const override { return false; }

    static void execute(const Tensor &input,
                        int peer,
                        infinicclComm_t communicator);

private:
    void *planned_meta_ = nullptr;
};

class Recv : public graph::GraphOperator {
public:
    Recv(Tensor output, int peer, infinicclComm_t communicator);
    ~Recv() override;

    void run() const override;
    bool is_device_graph_capture_safe() const override { return false; }

    static void execute(Tensor output,
                        int peer,
                        infinicclComm_t communicator);

private:
    void *planned_meta_ = nullptr;
};

void send(const Tensor &input, int peer, infinicclComm_t communicator);
void recv_(Tensor output, int peer, infinicclComm_t communicator);
Tensor recv(const Shape &shape,
            DataType dtype,
            Device device,
            int peer,
            infinicclComm_t communicator);

} // namespace infinicore::op::distributed
