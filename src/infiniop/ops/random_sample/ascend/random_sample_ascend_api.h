#ifndef __RANDOM_SAMPLE_ASCEND_API_H__
#define __RANDOM_SAMPLE_ASCEND_API_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "infiniop/operator.h"

struct InfiniopRandomSampleAscendDescriptor;
typedef struct InfiniopRandomSampleAscendDescriptor *infiniopRandomSampleAscendDescriptor_t;

infiniopStatus_t ascendCreateRandomSampleDescriptor(infiniopAscendHandle_t handle,
                                                    infiniopRandomSampleAscendDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t results,
                                                    infiniopTensorDescriptor_t probs);

infiniopStatus_t ascendGetRandomSampleWorkspaceSize(infiniopRandomSampleAscendDescriptor_t desc,
                                                    uint64_t *size);

infiniopStatus_t ascendRandomSample(infiniopRandomSampleAscendDescriptor_t desc,
                                    void *workspace,
                                    uint64_t workspace_size,
                                    void *result,
                                    void const *probs,
                                    float random_val,
                                    float topp,
                                    int topk,
                                    float temperature,
                                    void *stream);

infiniopStatus_t ascendDestroyRandomSampleDescriptor(infiniopRandomSampleAscendDescriptor_t desc);

#endif
