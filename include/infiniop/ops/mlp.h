#ifndef __INFINIOP_MLP_H__
#define __INFINIOP_MLP_H__

#include "../operator.h"
#include "matmul.h"
#include "swiglu.h"

typedef InfiniopDescriptor *infiniopMLPDescriptor_t;

__C __export infiniopStatus_t infiniopCreateMLPDescriptor(infiniopHandle_t handle,
                                                          infiniopMLPDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t y_desc,
                                                          infiniopTensorDescriptor_t x_desc,
                                                          infiniopTensorDescriptor_t w12_desc,
                                                          infiniopTensorDescriptor_t w3_desc,
                                                          float alpha,
                                                          char residual);

__C __export infiniopStatus_t infiniopGetMLPWorkspaceSize(infiniopMLPDescriptor_t desc, size_t *size);

__C __export infiniopStatus_t infiniopMLP(infiniopMLPDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *y,
                                          void const *x,
                                          void const *w12,
                                          void const *w3,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyMLPDescriptor(infiniopMLPDescriptor_t desc);
#endif
