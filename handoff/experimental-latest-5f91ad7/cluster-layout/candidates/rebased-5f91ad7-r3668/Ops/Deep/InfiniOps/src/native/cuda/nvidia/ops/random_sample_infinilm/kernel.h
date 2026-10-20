#ifndef INFINI_OPS_NVIDIA_RANDOM_SAMPLE_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_RANDOM_SAMPLE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/random_sample_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RandomSampleInfinilm, Device::Type::kNvidia>
    : public CudaRandomSampleInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRandomSampleInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaRandomSampleInfinilm;
};

}  // namespace infini::ops

#endif
