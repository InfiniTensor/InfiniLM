#ifndef INFINI_OPS_ILUVATAR_RANDOM_SAMPLE_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RANDOM_SAMPLE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/random_sample_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RandomSampleInfinilm, Device::Type::kIluvatar>
    : public CudaRandomSampleInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaRandomSampleInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaRandomSampleInfinilm;
};

}  // namespace infini::ops

#endif
