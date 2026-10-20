#ifndef INFINI_OPS_MOORE_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_MOORE_ROTARY_EMBEDDING_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/rotary_embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kMoore>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kMoore>> {
 public:
  using CudaRotaryEmbedding<Runtime<Device::Type::kMoore>>::CudaRotaryEmbedding;
};

}  // namespace infini::ops

#endif
