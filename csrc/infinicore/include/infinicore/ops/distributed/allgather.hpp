#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <infiniccl/infiniccl.h>
#include <vector>

namespace infinicore::op::distributed {

class AllGather : public graph::GraphOperator {
public:
    AllGather(Tensor output, const Tensor &input, infinicclComm_t communicator);
    ~AllGather();
    void run() const override;
    static void execute(Tensor output, const Tensor &input, infinicclComm_t communicator);

private:
    void *planned_meta_;
};

Tensor allgather(const Tensor &input, size_t world_size, infinicclComm_t communicator);
void allgather_(Tensor output, const Tensor &input, infinicclComm_t communicator);
Tensor allgatherv(const Tensor &input, const std::vector<size_t> &split_sizes, infinicclComm_t communicator);
void allgatherv_(Tensor output, const Tensor &input, const std::vector<size_t> &split_sizes, infinicclComm_t communicator);
std::vector<Tensor> allgatherv_many(const std::vector<Tensor> &inputs,
                                    const std::vector<size_t> &split_sizes,
                                    infinicclComm_t communicator);
void allgatherv_many_(const std::vector<Tensor> &outputs,
                      const std::vector<Tensor> &inputs,
                      const std::vector<size_t> &split_sizes,
                      infinicclComm_t communicator);

} // namespace infinicore::op::distributed
