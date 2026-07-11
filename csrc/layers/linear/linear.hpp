#pragma once

#include "../../engine/distributed/async_collective.hpp"
#include "../quantization/quantization.hpp"
#include "base_linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/ops.hpp"
#include <infiniccl.h>
#include <optional>
#include <vector>

namespace infinilm::nn {

class Linear : public BaseLinear {
public:
    // Without quantization (backward compat)
    Linear(size_t in_features, size_t out_features, bool bias,
           const infinicore::DataType &dtype = infinicore::DataType::F32,
           const infinicore::Device &device = infinicore::Device());

    // With quantization
    Linear(size_t in_features, size_t out_features,
           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
           bool bias = true,
           const infinicore::DataType &dtype = infinicore::DataType::F32,
           const infinicore::Device &device = infinicore::Device());

    infinicore::Tensor forward(infinicore::Tensor &input) const;
    std::string extra_repr() const;
};

class ColumnParallelLinear : public BaseLinear {
public:
    // Without quantization (backward compat)
    ColumnParallelLinear(size_t in_features, size_t out_features, bool bias,
                         const infinicore::DataType &dtype, const infinicore::Device &device,
                         infinicore::Size tp_rank = 0, infinicore::Size tp_size = 1,
                         int tp_num_heads = -1);

    // With quantization
    ColumnParallelLinear(size_t in_features, size_t out_features,
                         std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                         bool bias = true,
                         const infinicore::DataType &dtype = infinicore::DataType::F32,
                         const infinicore::Device &device = infinicore::Device(),
                         infinicore::Size tp_rank = 0, infinicore::Size tp_size = 1,
                         int tp_num_heads = -1);

    infinicore::Tensor forward(infinicore::Tensor &input) const;
    std::string extra_repr() const;

protected:
    infinicore::Size tp_rank_ = 0;
    infinicore::Size tp_size_ = 1;
};

struct RowParallelLinearOutput {
    infinicore::Tensor output;
    infinicore::Tensor allreduce_input;
    engine::distributed::AsyncCollectiveContext *async_context = nullptr;
    std::vector<engine::distributed::PendingCollective> pending_collectives;

    bool has_pending() const;
    void wait() const;
};

class RowParallelLinear : public BaseLinear {
public:
    // Without quantization (backward compat)
    RowParallelLinear(size_t in_features, size_t out_features, bool bias,
                      const infinicore::DataType &dtype, const infinicore::Device &device,
                      infinicore::Size tp_rank = 0, infinicore::Size tp_size = 1,
                      infinicclComm_t communicator = nullptr);

    // With quantization
    RowParallelLinear(size_t in_features, size_t out_features,
                      std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                      bool bias = true,
                      const infinicore::DataType &dtype = infinicore::DataType::F32,
                      const infinicore::Device &device = infinicore::Device(),
                      infinicore::Size tp_rank = 0, infinicore::Size tp_size = 1,
                      infinicclComm_t communicator = nullptr);

    infinicore::Tensor forward(infinicore::Tensor &input) const;
    RowParallelLinearOutput forward_async(infinicore::Tensor &input) const;
    std::string extra_repr() const;

protected:
    infinicore::Size tp_rank_ = 0;
    infinicore::Size tp_size_ = 1;
    infinicclComm_t communicator_;
};

} // namespace infinilm::nn

#include "fused_linear.hpp"

namespace infinilm::layers::linear {

using ReplicatedLinear = infinilm::nn::Linear;
using ColumnParallelLinear = infinilm::nn::ColumnParallelLinear;
using RowParallelLinear = infinilm::nn::RowParallelLinear;
using RowParallelLinearOutput = infinilm::nn::RowParallelLinearOutput;
using BaseLinear = infinilm::nn::BaseLinear;

} // namespace infinilm::layers::linear
