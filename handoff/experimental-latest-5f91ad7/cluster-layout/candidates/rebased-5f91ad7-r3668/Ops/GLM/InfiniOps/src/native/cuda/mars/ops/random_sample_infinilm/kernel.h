#ifndef INFINI_OPS_MARS_RANDOM_SAMPLE_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_RANDOM_SAMPLE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/random_sample_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RandomSampleInfinilm, Device::Type::kMars>
    : public CudaRandomSampleInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaRandomSampleInfinilm<
      Runtime<Device::Type::kMars>>::CudaRandomSampleInfinilm;
};

}  // namespace infini::ops

#endif
