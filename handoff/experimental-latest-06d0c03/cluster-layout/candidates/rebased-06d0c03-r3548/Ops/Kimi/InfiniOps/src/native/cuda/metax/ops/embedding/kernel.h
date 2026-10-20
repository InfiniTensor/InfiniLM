#ifndef INFINI_OPS_METAX_EMBEDDING_KERNEL_H_
#define INFINI_OPS_METAX_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kMetax>
    : public CudaEmbedding<Runtime<Device::Type::kMetax>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kMetax>>::CudaEmbedding;
};

}  // namespace infini::ops

#endif
