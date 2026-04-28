#ifndef __INFINIOP_UPSAMPLE_NEAREST_API_H__
#define __INFINIOP_UPSAMPLE_NEAREST_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopUpsampleNearestDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateUpsampleNearestDescriptor(infiniopHandle_t handle,
                                                                           infiniopUpsampleNearestDescriptor_t *desc_ptr,
                                                                           infiniopTensorDescriptor_t output,
                                                                           infiniopTensorDescriptor_t input);

__INFINI_C __export infiniStatus_t infiniopGetUpsampleNearestWorkspaceSize(infiniopUpsampleNearestDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopUpsampleNearest(infiniopUpsampleNearestDescriptor_t desc,
                                                           void *workspace,
                                                           size_t workspace_size,
                                                           void *output,
                                                           const void *input,
                                                           void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyUpsampleNearestDescriptor(infiniopUpsampleNearestDescriptor_t desc);

#endif // __INFINIOP_UPSAMPLE_NEAREST_API_H__
