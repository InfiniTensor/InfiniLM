#ifndef __INFINIOP_HYPOT_API_H__
#define __INFINIOP_HYPOT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopHypotDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHypotDescriptor(infiniopHandle_t handle,
                                                                 infiniopHypotDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t output,
                                                                 infiniopTensorDescriptor_t input_a,
                                                                 infiniopTensorDescriptor_t input_b);

__INFINI_C __export infiniStatus_t infiniopGetHypotWorkspaceSize(infiniopHypotDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHypot(infiniopHypotDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *output,
                                                 const void *input_a,
                                                 const void *input_b,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHypotDescriptor(infiniopHypotDescriptor_t desc);

#endif
