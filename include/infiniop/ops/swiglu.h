#ifndef __INFINIOP_SWIGLU_H__
#define __INFINIOP_SWIGLU_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopSwiGLUDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                                             infiniopSwiGLUDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t c_desc,
                                                             infiniopTensorDescriptor_t a_desc,
                                                             infiniopTensorDescriptor_t b_desc);

__C __export infiniopStatus_t infiniopSwiGLU(infiniopSwiGLUDescriptor_t desc,
                                             void *c,
                                             void const *a,
                                             void const *b,
                                             void *stream);

__C __export infiniopStatus_t infiniopDestroySwiGLUDescriptor(infiniopSwiGLUDescriptor_t desc);

#endif
