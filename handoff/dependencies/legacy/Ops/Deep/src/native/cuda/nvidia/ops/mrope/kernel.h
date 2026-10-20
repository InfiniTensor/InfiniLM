#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/mrope/kernel.h"
namespace infini::ops {
template<> class Operator<Mrope, Device::Type::kNvidia> : public CudaMrope<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaMrope<Runtime<Device::Type::kNvidia>>::CudaMrope;
};
}
