#ifndef INFINI_OPS_THEAD_EMBEDDING_KERNEL_H_
#define INFINI_OPS_THEAD_EMBEDDING_KERNEL_H_

#include "native/cuda/ops/embedding/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kThead>
    : public CudaEmbedding<Runtime<Device::Type::kThead>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kThead>>::CudaEmbedding;
};

}  // namespace infini::ops

#endif
