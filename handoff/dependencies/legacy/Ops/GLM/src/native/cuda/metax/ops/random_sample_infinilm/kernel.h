#ifndef INFINI_OPS_METAX_RANDOM_SAMPLE_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_RANDOM_SAMPLE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/random_sample_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RandomSampleInfinilm, Device::Type::kMetax>
    : public CudaRandomSampleInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaRandomSampleInfinilm<
      Runtime<Device::Type::kMetax>>::CudaRandomSampleInfinilm;
};

}  // namespace infini::ops

#endif
