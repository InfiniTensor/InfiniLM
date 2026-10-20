#pragma once

#include <cuda_runtime.h>

#include "flash.h"
#include "tensor.h"

namespace infini::ops::native_flash {

struct DeviceScope {
  int previous;
  explicit DeviceScope(int device);
  ~DeviceScope();
};

// Non-overlapping strided rows are valid regardless of logical axis order.
// Singleton axes may have zero stride; the innermost element stride is one.
bool plain_layout(Tensor t, int dimensions);
bool vector_aligned(Tensor t);

// Fixed FA specializations only: paged varlen and KV-cache share split-KV;
// packed, nonpaged varlen must retain the separate forward kernel family.
void launch_splitkv(infini_flash::Flash_fwd_params& p, cudaStream_t stream);
void launch_varlen(infini_flash::Flash_fwd_params& p, cudaStream_t stream);

}  // namespace infini::ops::native_flash
