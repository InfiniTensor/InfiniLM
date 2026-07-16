#pragma once

#include "../../device.hpp"
#include "../../graph/graph.hpp"
#include "../common/op.hpp"

#include <infiniccl/infiniccl.h>
#include <vector>

namespace infinicore::op::distributed {

class ReduceScatter : public graph::GraphOperator {
public:
    ReduceScatter(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator);
    ~ReduceScatter();
    void run() const override;
    static void execute(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator);

private:
    void *planned_meta_;
};

Tensor reduce_scatter(const Tensor &input, size_t world_size, infinicclRedOp_t op, infinicclComm_t communicator);
void reduce_scatter_(Tensor output, const Tensor &input, infinicclRedOp_t op, infinicclComm_t communicator);
Tensor reduce_scatterv(const Tensor &input,
                       const std::vector<size_t> &split_sizes,
                       size_t rank,
                       infinicclRedOp_t op,
                       infinicclComm_t communicator);
void reduce_scatterv_(Tensor output,
                      const Tensor &input,
                      const std::vector<size_t> &split_sizes,
                      infinicclRedOp_t op,
                      infinicclComm_t communicator);
std::vector<Tensor> reduce_scatterv_many(const std::vector<Tensor> &inputs,
                                         const std::vector<size_t> &split_sizes,
                                         size_t rank,
                                         infinicclRedOp_t op,
                                         infinicclComm_t communicator);
void reduce_scatterv_many_(const std::vector<Tensor> &outputs,
                           const std::vector<Tensor> &inputs,
                           const std::vector<size_t> &split_sizes,
                           infinicclRedOp_t op,
                           infinicclComm_t communicator);

} // namespace infinicore::op::distributed
