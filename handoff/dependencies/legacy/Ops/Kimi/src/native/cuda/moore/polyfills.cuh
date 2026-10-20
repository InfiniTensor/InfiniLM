#ifndef INFINI_OPS_MOORE_POLYFILLS_CUH_
#define INFINI_OPS_MOORE_POLYFILLS_CUH_

#include <type_traits>

// clang-format off
#include <musa_bf16.h>
#include <musa_fp16.h>
// clang-format on

namespace infini::ops {

template <typename T>
__device__ __forceinline__ T __hadd(const T& a, const T& b) {
  return a + b;
}

template <typename T>
__device__ __forceinline__ auto __high2bfloat16(const T& a) {
  return __float2bfloat16_rn(::__high2float(a));
}

template <typename T>
__device__ __forceinline__ T __hneg(const T& a) {
  return -a;
}

template <typename T>
__device__ __forceinline__ auto __low2bfloat16(const T& a) {
  return __float2bfloat16_rn(::__low2float(a));
}

template <typename T>
__device__ __forceinline__ T hrcp(const T& a) {
  return T(__frcp_rn(static_cast<float>(a)));
}

}  // namespace infini::ops

#define hrcp infini::ops::hrcp

#endif
