#ifndef __INFINIOP_CLIP_API_H__
#define __INFINIOP_CLIP_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopClipDescriptor_t;

__C __export infiniStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle,
                                                         infiniopClipDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y,
                                                         infiniopTensorDescriptor_t x,
                                                         infiniopTensorDescriptor_t min_val,
                                                         infiniopTensorDescriptor_t max_val);

__C __export infiniStatus_t infiniopGetClipWorkspaceSize(infiniopClipDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopClip(infiniopClipDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *y,
                                         const void *x,
                                         const void *min_val,
                                         const void *max_val,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc);

#endif
