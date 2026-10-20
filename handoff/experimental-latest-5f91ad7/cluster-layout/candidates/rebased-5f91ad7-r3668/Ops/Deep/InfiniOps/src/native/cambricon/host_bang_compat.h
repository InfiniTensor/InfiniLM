#ifndef INFINI_OPS_NATIVE_CAMBRICON_HOST_BANG_COMPAT_H_
#define INFINI_OPS_NATIVE_CAMBRICON_HOST_BANG_COMPAT_H_

// Neuware 6.0 leaves __BANG_HOSTDEVICE__ empty when its FP16/BF16 headers are
// parsed by a host C++ compiler. Consequently, the function definitions in
// those headers have external linkage and collide across translation units.
// Pre-include each header with the BANG annotations mapped to standard C++
// inline semantics. The vendor include guards then keep later cnrt/cnnl
// includes from emitting the non-inline definitions.
#if !defined(__BANGCC__) && !defined(__CUDACC__)
#pragma push_macro("__BANGCC__")
#pragma push_macro("__mlu_host__")
#pragma push_macro("__mlu_func__")
#define __BANGCC__ 1
#define __mlu_host__
#define __mlu_func__
#include <bang_fp16.h>
#pragma pop_macro("__mlu_func__")
#pragma pop_macro("__mlu_host__")
#pragma pop_macro("__BANGCC__")

#pragma push_macro("__BANGCC__")
#pragma push_macro("__mlu_host__")
#pragma push_macro("__mlu_func__")
#define __BANGCC__ 1
#define __mlu_host__ inline
#define __mlu_func__
#include <bang_bf16.h>
#pragma pop_macro("__mlu_func__")
#pragma pop_macro("__mlu_host__")
#pragma pop_macro("__BANGCC__")
#endif

#endif
