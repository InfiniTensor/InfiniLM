#ifndef INFINI_OPS_MOORE_EMBEDDING_KERNEL_H_
#define INFINI_OPS_MOORE_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kMoore>
    : public CudaEmbedding<Runtime<Device::Type::kMoore>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kMoore>>::CudaEmbedding;
};

}  // namespace infini::ops

#endif
