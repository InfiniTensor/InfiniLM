#include "core/inf_assert.h"
#include "infiniop/ops/topk.h"
#include "topk_info.h"
#include <cuda_runtime.h>
#include <iostream>

namespace infiniop {

struct infiniopTopKDescriptor : public infiniopDescriptor {
    TopKInfo info;
    infiniopTopKDescriptor(infiniopHandle_t handle,
                         infiniopTensorDescriptor_t input_desc,
                         infiniopTensorDescriptor_t output_val_desc,
                         infiniopTensorDescriptor_t output_ind_desc, int k)
        : infiniopDescriptor(handle, INFOP_OP_TOPK),
          info(input_desc, output_val_desc, output_ind_desc, k) {}
};

} // namespace infiniop

void infiniopCreateTopKDescriptor(infiniopHandle_t handle,
                                  infiniopTopKDescriptor_t *desc,
                                  infiniopTensorDescriptor_t input_desc,
                                  infiniopTensorDescriptor_t output_val_desc,
                                  infiniopTensorDescriptor_t output_ind_desc,
                                  int k) {
    *desc = new infiniop::infiniopTopKDescriptor(handle, input_desc,
                                               output_val_desc,
                                               output_ind_desc, k);
}

void infiniopDestroyTopKDescriptor(infiniopTopKDescriptor_t desc) {
    delete desc;
}

void infiniopGetTopKWorkspaceSize(infiniopTopKDescriptor_t desc, size_t *size) {
    *size = desc->info.workspace_size;
}

void infiniopTopK(infiniopTopKDescriptor_t desc, void *workspace,
                  size_t workspace_size, void *output_val, void *output_ind,
                  const void *input, infiniopStream_t stream) {
    desc->info.workspace = workspace;
    desc->info.workspace_size = workspace_size;
    desc->info.op(output_val, output_ind, input, stream);
} 