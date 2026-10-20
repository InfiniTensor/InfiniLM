#ifndef INFINI_OPS_MOORE_ROTARY_EMBEDDING_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_ROTARY_EMBEDDING_INFINILM_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/rotary_embedding_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbeddingInfinilm, Device::Type::kMoore>
    : public CudaRotaryEmbeddingInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaRotaryEmbeddingInfinilm<
      Runtime<Device::Type::kMoore>>::CudaRotaryEmbeddingInfinilm;
};

}  // namespace infini::ops

#endif
