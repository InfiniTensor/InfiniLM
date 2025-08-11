#ifndef __INFINIOP_KUNLUN_KERNEL_COMMON_H__
#define __INFINIOP_KUNLUN_KERNEL_COMMON_H__

// This header file will only be include by .xpu file
#include "xpu/runtime.h"
#include <xpu/kernel/xtdk.h>
#include <xpu/kernel/xtdk_bf16.h>
#include <xpu/kernel/xtdk_math.h>
#include <xpu/kernel/xtdk_simd.h>

namespace device::kunlun::kernel {

typedef struct _ptrdiff_t {
    int32_t value;   // 32 bit
    int32_t padding; // 32 bit
} _ptrdiff_t;

// same as ptrdiff
typedef struct _size_t {
    uint32_t value;
    uint32_t padding;
} _size_t;

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

/**
 * @brief Get index of broadcasted input
 * flat_index: flatten index of output tensor
 * ndim: dim of output tensor
 * broadcasted_strides: strides of output tensor
 * target_strides: strides of input tensor
 */
inline __device__ int indexToReducedOffset(
    int flat_index,                        // output flatten index
    int ndim,                              // output dims
    const _ptrdiff_t *broadcasted_strides, // output strides
    const _ptrdiff_t *target_strides) {    // strides of inputs

    int res = 0;
    for (int i = 0; i < ndim; ++i) {
        res += flat_index / broadcasted_strides[i].value * target_strides[i].value;
        flat_index %= broadcasted_strides[i].value;
    }
    return res;
}

/**
 * @brief Get real offset of input index
 * flat_index: flatten index input
 * ndim: dim of input tensor
 * shape: shape of input tensor
 * strides: strides of input tensor
 */
inline __device__ int indexToOffset(
    int flat_index,
    int ndim,
    const _size_t *shape,
    const _ptrdiff_t *strides) {

    int res = 0;
    for (int i = ndim; i-- > 0;) {
        res += (flat_index % shape[i].value) * strides[i].value;
        flat_index /= shape[i].value;
    }
    return res;
}

} // namespace device::kunlun::kernel

#endif // __INFINIOP_KUNLUN_KERNEL_COMMON_H__
// TODO: atomicAddF16
// TODO: atomicAddI8
