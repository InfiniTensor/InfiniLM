#ifndef __INFINIOP_RECIPROCAL_API_H__
#define __INFINIOP_RECIPROCAL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopReciprocalDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateReciprocalDescriptor(infiniopHandle_t handle,
                                                                      infiniopReciprocalDescriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t y,
                                                                      infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetReciprocalWorkspaceSize(infiniopReciprocalDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopReciprocal(infiniopReciprocalDescriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *y,
                                                      const void *x,
                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyReciprocalDescriptor(infiniopReciprocalDescriptor_t desc);

#endif
