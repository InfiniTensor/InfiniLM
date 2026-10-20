#ifndef INFINI_OPS_MOORE_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_

// clang-format off
#include <musa_runtime.h>
// clang-format on

// clang-format off
#include "native/cuda/moore/device_.h"
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/causal_softmax_infinilm/kernel.h"

namespace infini::ops {

struct MooreCausalSoftmaxInfinilmBackend : Runtime<Device::Type::kMoore> {
  // Moore's causal softmax kernel should not dispatch block sizes above 1024.
  static constexpr int max_block_size = 1024;
};

template <>
class Operator<CausalSoftmaxInfinilm, Device::Type::kMoore>
    : public CudaCausalSoftmaxInfinilm<MooreCausalSoftmaxInfinilmBackend> {
 public:
  using CudaCausalSoftmaxInfinilm<
      MooreCausalSoftmaxInfinilmBackend>::CudaCausalSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
