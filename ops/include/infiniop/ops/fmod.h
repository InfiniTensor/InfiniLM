#ifndef __INFINIOP_FMOD_API_H__
#define __INFINIOP_FMOD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFmodDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFmodDescriptor(infiniopHandle_t handle,
                                                                infiniopFmodDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t c,
                                                                infiniopTensorDescriptor_t a,
                                                                infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetFmodWorkspaceSize(infiniopFmodDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopFmod(infiniopFmodDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *c,
                                                const void *a,
                                                const void *b,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFmodDescriptor(infiniopFmodDescriptor_t desc);

#endif
