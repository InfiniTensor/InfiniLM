#ifndef __INFINIOP_KUNLUN_COMMON_H__
#define __INFINIOP_KUNLUN_COMMON_H__

// This header file will only be include by .xpu file
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_math.h"
#include "xpu/kernel/xtdk_simd.h"
#include "xpu/runtime.h"

// Get mask for vload_lm_ func
// 0 - i bit 1, others 0
static inline __device__ float lowerBitMask(int i) {
    return (1 << (i + 1)) - 1;
}

// Atomic add for reduce
static inline __device__ void atomic_add(__shared_ptr__ float *ptr, float value) {
    int fail = 1;
    while (fail) {
        float a = SM2REG_atomic(ptr);
        a = a + value;
        fail = REG2SM_atomic(ptr, a);
    }
}

#endif
