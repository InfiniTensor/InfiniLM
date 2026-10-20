#ifndef INFINI_OPS_ILUVATAR_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ILUVATAR_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kIluvatar>
    : public CudaEmbedding<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kIluvatar>>::CudaEmbedding;
};

}  // namespace infini::ops

#endif
