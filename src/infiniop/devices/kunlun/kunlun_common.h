#ifndef __INFINIOP_KUNLUN_COMMON_H__
#define __INFINIOP_KUNLUN_COMMON_H__

// This header file will only be include by .xpu file
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_math.h"
#include "xpu/kernel/xtdk_simd.h"
#include "xpu/runtime.h"

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

// TODO: atomicAddF16
// TODO: atomicAddI8
#endif
