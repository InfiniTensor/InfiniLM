#ifndef __INFINIOP_FMIN_H__
#define __INFINIOP_FMIN_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFminDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFminDescriptor(infiniopHandle_t handle,
                                                                infiniopFminDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t c,
                                                                infiniopTensorDescriptor_t a,
                                                                infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetFminWorkspaceSize(infiniopFminDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopFmin(infiniopFminDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *c,
                                                const void *a,
                                                const void *b,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFminDescriptor(infiniopFminDescriptor_t desc);

#endif
