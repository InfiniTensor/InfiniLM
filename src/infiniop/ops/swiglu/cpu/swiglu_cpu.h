#ifndef __INFINIOP_SWIGLU_CPU_H__
#define __INFINIOP_SWIGLU_CPU_H__

#include "./swiglu_cpu_api.h"

typedef struct SwiGLUCpuDescriptor {
    infiniDevice_t device;
    infiniDtype_t dtype;
    size_t n, d;
    ptrdiff_t
        s_no, // n stride of out
        s_do, // d stride of out
        s_ng, // n stride of gate
        s_dg, // d stride of gate
        s_nu, // n stride of up
        s_du; // d stride of up
} SwiGLUCpuDescriptor;

#endif // __INFINIOP_SWIGLU_CPU_H__
