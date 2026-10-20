#ifndef INFINI_OPS_CAMBRICON_DATA_TYPE__H_
#define INFINI_OPS_CAMBRICON_DATA_TYPE__H_

#include <infini/rt/cambricon/data_type_.h>
// Native arithmetic is confined to genuine BANG compilation. The public RT
// type map uses the same complete storage types in host and native objects.
#if defined(__BANGCC__)
#include <bang_bf16.h>
#include <bang_fp16.h>

namespace infini::ops::cambricon {
template <typename StorageT>
struct NativeScalar {
  using type = StorageT;
};
template <>
struct NativeScalar<infini::rt::Float16> {
  using type = __half;
};
template <>
struct NativeScalar<infini::rt::BFloat16> {
  using type = __bang_bfloat16;
};
template <typename StorageT>
using NativeScalarType = typename NativeScalar<StorageT>::type;
}  // namespace infini::ops::cambricon
#endif

#endif
