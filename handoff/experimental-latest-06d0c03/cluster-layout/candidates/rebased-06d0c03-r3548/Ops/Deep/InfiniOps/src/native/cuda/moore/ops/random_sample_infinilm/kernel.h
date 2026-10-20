#ifndef INFINI_OPS_MOORE_RANDOM_SAMPLE_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_RANDOM_SAMPLE_INFINILM_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/random_sample_infinilm/kernel.h"

namespace infini::ops {

struct MooreRandomSampleInfinilmBackend : Runtime<Device::Type::kMoore> {
  static constexpr int max_block_size = 1024;
};

template <>
class Operator<RandomSampleInfinilm, Device::Type::kMoore>
    : public CudaRandomSampleInfinilm<MooreRandomSampleInfinilmBackend> {
 public:
  using CudaRandomSampleInfinilm<
      MooreRandomSampleInfinilmBackend>::CudaRandomSampleInfinilm;
};

}  // namespace infini::ops

#endif
