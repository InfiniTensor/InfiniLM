#ifndef __INFINIOP_TANHSHRINK_API_H__
#define __INFINIOP_TANHSHRINK_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTanhshrinkDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTanhshrinkDescriptor(infiniopHandle_t handle,
                                                                      infiniopTanhshrinkDescriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t output,
                                                                      infiniopTensorDescriptor_t intput);

__INFINI_C __export infiniStatus_t infiniopGetTanhshrinkWorkspaceSize(infiniopTanhshrinkDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTanhshrink(infiniopTanhshrinkDescriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *output,
                                                      const void *intput,
                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTanhshrinkDescriptor(infiniopTanhshrinkDescriptor_t desc);

#endif
