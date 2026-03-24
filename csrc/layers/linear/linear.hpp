#pragma once
#include "fused_linear.hpp"

namespace infinilm::layers::linear {
using QKVParallelLinear = infinilm::layers::linear::QKVParallelLinear;
using ReplicatedLinear = infinicore::nn::Linear;
using ColumnParallelLinear = infinicore::nn::ColumnParallelLinear;
using RowParallelLinear = infinicore::nn::RowParallelLinear;
using GateUpParallelLinear = infinilm::layers::linear::GateUpParallelLinear;
} // namespace infinilm::layers::linear
