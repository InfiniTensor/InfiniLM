#ifndef __INFINIOP_API_H__
#define __INFINIOP_API_H__

#include "infiniop/handle.h"
#include "infiniop/ops/add.h"
#include "infiniop/ops/attention.h"
#include "infiniop/ops/causal_softmax.h"
#include "infiniop/ops/clip.h"
#include "infiniop/ops/conv.h"
#include "infiniop/ops/gemm.h"
#include "infiniop/ops/mul.h"
#include "infiniop/ops/random_sample.h"
#include "infiniop/ops/rearrange.h"
#include "infiniop/ops/relu.h"
#include "infiniop/ops/rms_norm.h"
#include "infiniop/ops/rope.h"
#include "infiniop/ops/sub.h"
#include "infiniop/ops/swiglu.h"
#include "infiniop/ops/topk.h"
#include "infiniop/ops/moe_dispatch.h"
#include "infiniop/ops/moe_combine.h"
#include "infiniop/tensor_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INFINIOP_TOPK_STRATEGY_DEEPSEEK_V3 0
#define INFINIOP_TOPK_STRATEGY_STANDARD_SOFTMAX 1

__C infiniStatus_t
infiniopCreateTopKDescriptor(infiniopHandle_t handle,
                             infiniopTopKDescriptor_t *desc_ptr,
                             infiniopTensorDescriptor_t input_desc,
                             infiniopTensorDescriptor_t output_val_desc,
                             infiniopTensorDescriptor_t output_ind_desc,
                             infiniopTensorDescriptor_t bias_desc, int k,
                             int strategy, int n_group,
                             int topk_group);

__C infiniStatus_t infiniopDestroyTopKDescriptor(infiniopTopKDescriptor_t desc);

__C size_t infiniopGetTopKWorkspaceSize(infiniopTopKDescriptor_t desc);

__C infiniStatus_t infiniopTopKCalculate(infiniopTopKDescriptor_t desc,
                                       const void *input, void *output_val,
                                       void *output_ind, const void *bias,
                                       void *workspace, void *stream);

#ifdef __cplusplus
}
#endif

#endif // __INFINIOP_API_H__
