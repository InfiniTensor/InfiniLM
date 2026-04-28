#ifndef __INFINIOP_LOGADDEXP2_API_H__
#define __INFINIOP_LOGADDEXP2_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogAddExp2Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLogAddExp2Descriptor(infiniopHandle_t handle,
                                                                      infiniopLogAddExp2Descriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t c,
                                                                      infiniopTensorDescriptor_t a,
                                                                      infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetLogAddExp2WorkspaceSize(infiniopLogAddExp2Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLogAddExp2(infiniopLogAddExp2Descriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *c,
                                                      const void *a,
                                                      const void *b,
                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLogAddExp2Descriptor(infiniopLogAddExp2Descriptor_t desc);

#endif // __INFINIOP_LOGADDEXP2_API_H__
