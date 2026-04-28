#ifndef __INFINIOP_MASKED_SELECT_API_H__
#define __INFINIOP_MASKED_SELECT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMaskedSelectDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateMaskedSelectDescriptor(
    infiniopHandle_t handle,
    infiniopMaskedSelectDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t mask_desc);

__INFINI_C __export infiniStatus_t infiniopGetMaskedSelectWorkspaceSize(infiniopMaskedSelectDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopMaskedSelect(
    infiniopMaskedSelectDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    const void *input,
    const bool *mask,
    void **data_ptr,
    size_t *dlen,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMaskedSelectDescriptor(infiniopMaskedSelectDescriptor_t desc);

#endif
