#ifndef __INFINIOP_UPSAMPLE_BILINEAR_API_H__
#define __INFINIOP_UPSAMPLE_BILINEAR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopUpsampleBilinearDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateUpsampleBilinearDescriptor(infiniopHandle_t handle,
                                                                            infiniopUpsampleBilinearDescriptor_t *desc_ptr,
                                                                            infiniopTensorDescriptor_t output,
                                                                            infiniopTensorDescriptor_t input,
                                                                            int align_corners);

__INFINI_C __export infiniStatus_t infiniopGetUpsampleBilinearWorkspaceSize(infiniopUpsampleBilinearDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopUpsampleBilinear(infiniopUpsampleBilinearDescriptor_t desc,
                                                            void *workspace,
                                                            size_t workspace_size,
                                                            void *output,
                                                            const void *input,
                                                            void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyUpsampleBilinearDescriptor(infiniopUpsampleBilinearDescriptor_t desc);

#endif // __INFINIOP_UPSAMPLE_BILINEAR_API_H__
