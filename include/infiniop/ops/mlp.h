#ifndef __INFINIOP_MLP_API_H__
#define __INFINIOP_MLP_API_H__

#include "../operator_descriptor.h"
#include "gemm.h"
#include "swiglu.h"

typedef struct InfiniopDescriptor *infiniopMLPDescriptor_t;

__C __export infiniStatus_t infiniopCreateMLPDescriptor(infiniopHandle_t handle,
                                                        infiniopMLPDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y_desc,
                                                        infiniopTensorDescriptor_t x_desc,
                                                        infiniopTensorDescriptor_t w12_desc,
                                                        infiniopTensorDescriptor_t w3_desc,
                                                        float alpha,
                                                        char residual);

__C __export infiniStatus_t infiniopGetMLPWorkspaceSize(infiniopMLPDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMLP(infiniopMLPDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        const void *w12,
                                        const void *w3,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyMLPDescriptor(infiniopMLPDescriptor_t desc);
#endif
