#pragma once
#include "fused_linear.hpp"
#ifdef INFINILM_ENABLE_INFINIOPS
#include "../../backends/operators/operators.hpp"
#endif

namespace infinilm::layers::linear {

using QKVParallelLinear = infinilm::layers::linear::QKVParallelLinear;
#ifdef INFINILM_ENABLE_INFINIOPS
using ReplicatedLinear = infinilm::backends::ops::ReplicatedLinear;
using ColumnParallelLinear = infinilm::backends::ops::ColumnParallelLinear;
using RowParallelLinear = infinilm::backends::ops::RowParallelLinear;
#else
using ReplicatedLinear = infinicore::nn::Linear;
using ColumnParallelLinear = infinicore::nn::ColumnParallelLinear;
using RowParallelLinear = infinicore::nn::RowParallelLinear;
#endif
using GateUpParallelLinear = infinilm::layers::linear::GateUpParallelLinear;

} // namespace infinilm::layers::linear
