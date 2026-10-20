#ifndef INFINI_OPS_MARS_EMBEDDING_KERNEL_H_
#define INFINI_OPS_MARS_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kMars>
    : public CudaEmbedding<Runtime<Device::Type::kMars>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kMars>>::CudaEmbedding;
};

}  // namespace infini::ops

#endif
