#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <infiniccl/infiniccl.h>

namespace infinicore::op::distributed {
class AllReduce : public graph::GraphOperator {
public:
    AllReduce(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator);
    ~AllReduce();
    void run() const override;
    bool is_device_graph_capture_safe() const override;
    static void execute(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator);

private:
    void *planned_meta_;
    bool device_graph_capture_safe_;
};

Tensor allreduce(const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator);
void allreduce_(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator);

} // namespace infinicore::op::distributed
