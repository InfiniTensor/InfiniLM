#ifndef __INFINIOP_KUNLUN_DTYPE_H__
#define __INFINIOP_KUNLUN_DTYPE_H__

#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_math.h"
#include "xpu/kernel/xtdk_simd.h"
#include "xpu/runtime.h"

// kunlun ptrdiff_t* is used to save ptrdiff_t array
// copied from host
typedef struct _ptrdiff_t {
    long value;   // 32 bit
    long padding; // 32 bit
} _ptrdiff_t;

// same as ptrdiff
typedef struct _size_t {
    size_t value;
    size_t padding;
} _size_t;

#endif
