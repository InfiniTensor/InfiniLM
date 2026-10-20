#include "flash_common.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "native/cuda/nvidia/ops/infini_flash_cuda.h"

namespace infini::ops::native_flash {

DeviceScope::DeviceScope(int device) {
  INFINI_FLASH_CUDA_CHECK(cudaGetDevice(&previous));
  INFINI_FLASH_CUDA_CHECK(cudaSetDevice(device));
}

DeviceScope::~DeviceScope() { cudaSetDevice(previous); }

bool plain_layout(Tensor t, int dimensions) {
  if (dimensions < 1 || dimensions > 4 || t.ndim() != dimensions ||
      t.stride(-1) != 1)
    return false;
  std::array<int, 4> axes{};
  int count = 0;
  for (int axis = 0; axis < dimensions; ++axis) {
    if (t.size(axis) <= 0 ||
        t.size(axis) > std::numeric_limits<int>::max() - 127 ||
        t.stride(axis) < 0)
      return false;
    // A singleton is never indexed away from zero, including LM unsqueeze
    // views with stride zero. Other zero-stride axes must still be rejected.
    if (t.size(axis) != 1) axes[count++] = axis;
  }
  std::sort(axes.begin(), axes.begin() + count,
            [&](int a, int b) { return t.stride(a) < t.stride(b); });
  std::size_t extent = 1;
  for (int index = 0; index < count; ++index) {
    const auto axis = axes[index];
    const auto size = static_cast<std::size_t>(t.size(axis));
    const auto stride = static_cast<std::size_t>(t.stride(axis));
    if (stride < extent ||
        (size - 1) >
            (std::numeric_limits<std::size_t>::max() - extent) / stride)
      return false;
    extent += (size - 1) * stride;
  }
  return true;
}

bool vector_aligned(Tensor t) {
  if (t.data() && reinterpret_cast<std::uintptr_t>(t.data()) % 16) return false;
  for (int axis = 0; axis < t.ndim() - 1; ++axis)
    if (t.size(axis) > 1 && t.stride(axis) % 8) return false;
  return true;
}

#define DECLARE_FLASH_LAUNCHES(prefix, dimension)                         \
  void prefix##_##dimension##_fp16_false(infini_flash::Flash_fwd_params&, \
                                         cudaStream_t);                   \
  void prefix##_##dimension##_fp16_true(infini_flash::Flash_fwd_params&,  \
                                        cudaStream_t);                    \
  void prefix##_##dimension##_bf16_false(infini_flash::Flash_fwd_params&, \
                                         cudaStream_t);                   \
  void prefix##_##dimension##_bf16_true(infini_flash::Flash_fwd_params&,  \
                                        cudaStream_t)

DECLARE_FLASH_LAUNCHES(launch, 32);
DECLARE_FLASH_LAUNCHES(launch, 64);
DECLARE_FLASH_LAUNCHES(launch, 96);
DECLARE_FLASH_LAUNCHES(launch, 128);
DECLARE_FLASH_LAUNCHES(launch, 192);
DECLARE_FLASH_LAUNCHES(launch, 256);
DECLARE_FLASH_LAUNCHES(launch_varlen, 32);
DECLARE_FLASH_LAUNCHES(launch_varlen, 64);
DECLARE_FLASH_LAUNCHES(launch_varlen, 96);
DECLARE_FLASH_LAUNCHES(launch_varlen, 128);
DECLARE_FLASH_LAUNCHES(launch_varlen, 192);
DECLARE_FLASH_LAUNCHES(launch_varlen, 256);
#undef DECLARE_FLASH_LAUNCHES

#define FLASH_LAUNCH_CASE(prefix, dimension)          \
  case dimension:                                     \
    if (p.is_bf16) {                                  \
      if (p.is_causal)                                \
        prefix##_##dimension##_bf16_true(p, stream);  \
      else                                            \
        prefix##_##dimension##_bf16_false(p, stream); \
    } else {                                          \
      if (p.is_causal)                                \
        prefix##_##dimension##_fp16_true(p, stream);  \
      else                                            \
        prefix##_##dimension##_fp16_false(p, stream); \
    }                                                 \
    break

void launch_splitkv(infini_flash::Flash_fwd_params& p, cudaStream_t stream) {
  switch (p.d_rounded) {
    FLASH_LAUNCH_CASE(launch, 32);
    FLASH_LAUNCH_CASE(launch, 64);
    FLASH_LAUNCH_CASE(launch, 96);
    FLASH_LAUNCH_CASE(launch, 128);
    FLASH_LAUNCH_CASE(launch, 192);
    FLASH_LAUNCH_CASE(launch, 256);
    default:
      throw std::logic_error("unsupported native Flash head dimension");
  }
}

void launch_varlen(infini_flash::Flash_fwd_params& p, cudaStream_t stream) {
  switch (p.d_rounded) {
    FLASH_LAUNCH_CASE(launch_varlen, 32);
    FLASH_LAUNCH_CASE(launch_varlen, 64);
    FLASH_LAUNCH_CASE(launch_varlen, 96);
    FLASH_LAUNCH_CASE(launch_varlen, 128);
    FLASH_LAUNCH_CASE(launch_varlen, 192);
    FLASH_LAUNCH_CASE(launch_varlen, 256);
    default:
      throw std::logic_error("unsupported native Flash head dimension");
  }
}
#undef FLASH_LAUNCH_CASE

}  // namespace infini::ops::native_flash
