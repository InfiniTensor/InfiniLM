#ifndef INFINI_OPS_METAX_ROTARY_EMBEDDING_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_ROTARY_EMBEDDING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/rotary_embedding_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbeddingInfinilm, Device::Type::kMetax>
    : public CudaRotaryEmbeddingInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaRotaryEmbeddingInfinilm<
      Runtime<Device::Type::kMetax>>::CudaRotaryEmbeddingInfinilm;
};

}  // namespace infini::ops

#endif
