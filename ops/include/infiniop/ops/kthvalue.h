#ifndef __INFINIOP_KTHVALUE_API_H__
#define __INFINIOP_KTHVALUE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopKthvalueDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateKthvalueDescriptor(infiniopHandle_t handle,
                                                                    infiniopKthvalueDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t values,
                                                                    infiniopTensorDescriptor_t indices,
                                                                    infiniopTensorDescriptor_t input,
                                                                    int k,
                                                                    int dim,
                                                                    int keepdim);

__INFINI_C __export infiniStatus_t infiniopGetKthvalueWorkspaceSize(infiniopKthvalueDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopKthvalue(infiniopKthvalueDescriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *values,
                                                    void *indices,
                                                    const void *input,
                                                    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyKthvalueDescriptor(infiniopKthvalueDescriptor_t desc);

#endif // __INFINIOP_KTHVALUE_API_H__
