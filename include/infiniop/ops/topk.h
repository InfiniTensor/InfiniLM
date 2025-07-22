// Content for topk.h
#ifndef __INFINIOP_TOPK_API_H__
#define __INFINIOP_TOPK_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTopKDescriptor_t;

__C __export infiniStatus_t infiniopCreateTopKDescriptor(
    infiniopHandle_t handle, infiniopTopKDescriptor_t *desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_val_desc,
    infiniopTensorDescriptor_t output_ind_desc,
    infiniopTensorDescriptor_t bias_desc, int k, int strategy,
    int n_group, int topk_group);

__C __export infiniStatus_t infiniopDestroyTopKDescriptor(infiniopTopKDescriptor_t desc);

__C __export size_t
infiniopGetTopKWorkspaceSize(infiniopTopKDescriptor_t desc);

__C __export infiniStatus_t
infiniopTopK(infiniopTopKDescriptor_t desc, const void *input, void *output_val,
           void *output_ind, const void *bias, void *workspace, void *stream);

#endif 