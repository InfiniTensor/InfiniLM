#ifndef INFINI_OPS_MOORE_PAGED_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_PAGED_CACHING_INFINILM_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/paged_caching_infinilm/kernel.h"

namespace infini::ops {

struct MoorePagedCachingInfinilmBackend : Runtime<Device::Type::kMoore> {
  static constexpr int max_block_size = 1024;
};

template <>
class Operator<PagedCachingInfinilm, Device::Type::kMoore>
    : public CudaPagedCachingInfinilm<MoorePagedCachingInfinilmBackend> {
 public:
  using CudaPagedCachingInfinilm<
      MoorePagedCachingInfinilmBackend>::CudaPagedCachingInfinilm;
};

}  // namespace infini::ops

#endif
