#ifndef __INFINIOP_PIXEL_SHUFFLE_API_H__
#define __INFINIOP_PIXEL_SHUFFLE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopPixelShuffleDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePixelShuffleDescriptor(infiniopHandle_t handle,
                                                                        infiniopPixelShuffleDescriptor_t *desc_ptr,
                                                                        infiniopTensorDescriptor_t y,
                                                                        infiniopTensorDescriptor_t x,
                                                                        int upscale_factor);

__INFINI_C __export infiniStatus_t infiniopGetPixelShuffleWorkspaceSize(infiniopPixelShuffleDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopPixelShuffle(infiniopPixelShuffleDescriptor_t desc,
                                                        void *workspace,
                                                        size_t workspace_size,
                                                        void *y,
                                                        const void *x,
                                                        void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPixelShuffleDescriptor(infiniopPixelShuffleDescriptor_t desc);

#endif
