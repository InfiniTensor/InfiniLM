#ifndef __INFINIOP_INDEX_ADD_API_H__
#define __INFINIOP_INDEX_ADD_API_H__
#include "../operator_descriptor.h"
#include <stdint.h>

typedef struct InfiniopDescriptor *infiniopIndexAddDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateIndexAddDescriptor(infiniopHandle_t handle,
                                                                    infiniopIndexAddDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t output,
                                                                    infiniopTensorDescriptor_t input,
                                                                    int64_t dim,
                                                                    infiniopTensorDescriptor_t index,
                                                                    infiniopTensorDescriptor_t source,
                                                                    float alpha);

__INFINI_C __export infiniStatus_t infiniopGetIndexAddWorkspaceSize(infiniopIndexAddDescriptor_t desc, size_t *size);
__INFINI_C __export infiniStatus_t infiniopIndexAdd(infiniopIndexAddDescriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *output,
                                                    const void *input,
                                                    const void *index,
                                                    const void *source,
                                                    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyIndexAddDescriptor(infiniopIndexAddDescriptor_t desc);

#endif
