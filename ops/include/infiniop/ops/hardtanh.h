#ifndef __INFINIOP_HARDTANH_API_H__
#define __INFINIOP_HARDTANH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopHardTanhDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHardTanhDescriptor(infiniopHandle_t handle,
                                                                    infiniopHardTanhDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t output,
                                                                    infiniopTensorDescriptor_t input,
                                                                    float min_val,
                                                                    float max_val);

__INFINI_C __export infiniStatus_t infiniopGetHardTanhWorkspaceSize(infiniopHardTanhDescriptor_t desc,
                                                                    size_t *size);

__INFINI_C __export infiniStatus_t infiniopHardTanh(infiniopHardTanhDescriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *output,
                                                    const void *input,
                                                    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHardTanhDescriptor(infiniopHardTanhDescriptor_t desc);

#endif
