#ifndef __INFINIOP_LOGCUMSUMEXP_API_H__
#define __INFINIOP_LOGCUMSUMEXP_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogCumSumExpDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLogCumSumExpDescriptor(infiniopHandle_t handle,
                                                                        infiniopLogCumSumExpDescriptor_t *desc_ptr,
                                                                        infiniopTensorDescriptor_t y,
                                                                        infiniopTensorDescriptor_t x,
                                                                        int axis,
                                                                        int exclusive,
                                                                        int reverse);

/* 获取执行 LogCumSumExp 所需的临时空间大小 */
__INFINI_C __export infiniStatus_t infiniopGetLogCumSumExpWorkspaceSize(infiniopLogCumSumExpDescriptor_t desc,
                                                                        size_t *size);

__INFINI_C __export infiniStatus_t infiniopLogCumSumExp(infiniopLogCumSumExpDescriptor_t desc,
                                                        void *workspace,
                                                        size_t workspace_size,
                                                        void *y,
                                                        const void *x,
                                                        void *stream);

/* 销毁描述符 */
__INFINI_C __export infiniStatus_t infiniopDestroyLogCumSumExpDescriptor(infiniopLogCumSumExpDescriptor_t desc);

#endif
