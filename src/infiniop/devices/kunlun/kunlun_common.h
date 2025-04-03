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

#endif