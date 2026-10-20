#ifndef INFINI_OPS_NVIDIA_EMBEDDING_KERNEL_H_
#define INFINI_OPS_NVIDIA_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kNvidia>
    : public CudaEmbedding<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kNvidia>>::CudaEmbedding;
  bool capture_safe() const override { return !max_norm_.has_value(); }
};

}  // namespace infini::ops

#endif
