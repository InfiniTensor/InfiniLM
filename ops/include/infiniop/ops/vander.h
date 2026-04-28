#ifndef __INFINIOP_VANDER_API_H__
#define __INFINIOP_VANDER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopVanderDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateVanderDescriptor(infiniopHandle_t handle,
                                                                  infiniopVanderDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t output,
                                                                  infiniopTensorDescriptor_t input,
                                                                  int N,
                                                                  int increasing);

__INFINI_C __export infiniStatus_t infiniopGetVanderWorkspaceSize(infiniopVanderDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopVander(infiniopVanderDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *output,
                                                  const void *input,
                                                  void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyVanderDescriptor(infiniopVanderDescriptor_t desc);

#endif // __INFINIOP_VANDER_API_H__
