#pragma once

#include "../quantization/quantization.hpp"
#include "base_linear.hpp"
#include "../../weights/weights.hpp"
#include <infiniccl/infiniccl.h>
#include <optional>

namespace infinilm::nn {

class Linear : public BaseLinear {
public:
    // Runtime-first supply; output dtype defaults to the parameter dtype.
    Linear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features, bool bias,
           const infinilm::runtime::DataType &dtype = infinilm::runtime::DataType::kFloat32,
           std::optional<infinilm::runtime::DataType> output_dtype = std::nullopt);

    Linear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features,
           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
           bool bias = true,
           const infinilm::runtime::DataType &dtype = infinilm::runtime::DataType::kFloat32);

    std::string extra_repr() const;
};

class ColumnParallelLinear : public BaseLinear {
public:
    // Runtime-first supply.
    ColumnParallelLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features, bool bias,
                         const infinilm::runtime::DataType &dtype,
                         infinilm::runtime::Size tp_rank = 0, infinilm::runtime::Size tp_size = 1,
                         int tp_num_heads = -1);

    ColumnParallelLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features,
                         std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                         bool bias = true,
                         const infinilm::runtime::DataType &dtype = infinilm::runtime::DataType::kFloat32,
                         infinilm::runtime::Size tp_rank = 0, infinilm::runtime::Size tp_size = 1,
                         int tp_num_heads = -1);

    std::string extra_repr() const;

protected:
    infinilm::runtime::Size tp_rank_ = 0;
    infinilm::runtime::Size tp_size_ = 1;
};

class RowParallelLinear : public BaseLinear {
public:
    // Runtime-first supply.
    RowParallelLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features, bool bias,
                      const infinilm::runtime::DataType &dtype,
                      infinilm::runtime::Size tp_rank = 0, infinilm::runtime::Size tp_size = 1,
                      infinicclComm_t communicator = nullptr);

    RowParallelLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features,
                      std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                      bool bias = true,
                      const infinilm::runtime::DataType &dtype = infinilm::runtime::DataType::kFloat32,
                      infinilm::runtime::Size tp_rank = 0, infinilm::runtime::Size tp_size = 1,
                      infinicclComm_t communicator = nullptr);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) const;
    std::string extra_repr() const;

protected:
    infinilm::runtime::Size tp_rank_ = 0;
    infinilm::runtime::Size tp_size_ = 1;
    infinicclComm_t communicator_;
};

} // namespace infinilm::nn

#include "fused_linear.hpp"

namespace infinilm::layers::linear {

// Pure metadata preparation for an explicit final dense-head request.
// Absence preserves the existing platform/dtype behavior; no provider is selected here.
std::optional<infinilm::runtime::DataType> prepare_lm_head_output_dtype(
    const infinilm::runtime::Device &device, infinilm::runtime::DataType weight_dtype,
    std::optional<infinilm::runtime::DataType> requested_output_dtype);

using ReplicatedLinear = infinilm::nn::Linear;
using ColumnParallelLinear = infinilm::nn::ColumnParallelLinear;
using RowParallelLinear = infinilm::nn::RowParallelLinear;
using BaseLinear = infinilm::nn::BaseLinear;

} // namespace infinilm::layers::linear
