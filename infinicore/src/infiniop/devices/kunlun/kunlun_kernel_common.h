#ifndef __INFINIOP_KUNLUN_KERNEL_COMMON_H__
#define __INFINIOP_KUNLUN_KERNEL_COMMON_H__

// This header file will only be include by .xpu file
#include "kunlun_kernel_dtype.h"
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_math.h"
#include "xpu/kernel/xtdk_simd.h"
#include "xpu/runtime.h"

namespace device::kunlun::kernel {
// Get mask for kunlun xpu 512bit register calculation
// if data is not enough to 512bit, padding zero and use
// mask to identify real data
// 0 - i bit 1, others 0
inline __device__ float lowerBitMask(int i) {
    return (1 << (i + 1)) - 1;
}

// Atomic add for reduce
inline __device__ void atomicAddF32(__shared_ptr__ float *ptr, float value) {
    int success = 1;
    while (success) {
        // SM2REG read 32bit data to register
        float a = SM2REG_atomic(ptr);
        a = a + value;
        success = REG2SM_atomic(ptr, a);
    }
}

inline __device__ size_t indexToReducedOffset(
    size_t flat_index,
    size_t ndim,
    const _ptrdiff_t *broadcasted_strides,
    const _ptrdiff_t *target_strides) {

    size_t res = 0;
    for (size_t i = 0; i < ndim; ++i) {
        res += flat_index / broadcasted_strides[i].value * target_strides[i].value;
        flat_index %= broadcasted_strides[i].value;
        mfence();
    }
    return res;
}

inline __device__ size_t indexToOffset(
    size_t flat_index,
    size_t ndim,
    const _size_t *shape,
    const _ptrdiff_t *strides) {

    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i].value) * strides[i].value;
        flat_index /= shape[i].value;
        mfence();
    }
    return res;
}

} // namespace device::kunlun::kernel
// TODO: atomicAddF16
// TODO: atomicAddI8
#endif
