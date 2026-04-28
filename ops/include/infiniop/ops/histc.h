#ifndef __INFINIOP_HISTC_API_H__
#define __INFINIOP_HISTC_API_H__

#include "../operator_descriptor.h"
#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopHistcDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHistcDescriptor(infiniopHandle_t handle,
                                                                 infiniopHistcDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x,
                                                                 int64_t bins,
                                                                 double min_val,
                                                                 double max_val);

__INFINI_C __export infiniStatus_t infiniopGetHistcWorkspaceSize(infiniopHistcDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHistc(infiniopHistcDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *y,
                                                 const void *x,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHistcDescriptor(infiniopHistcDescriptor_t desc);

#endif
