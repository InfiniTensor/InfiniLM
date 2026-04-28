#ifndef __INFINIOP_QUICKGELU_API_H__
#define __INFINIOP_QUICKGELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopQuickGeluDescriptor_t;

/**
 * Create QuickGELU descriptor
 * y = x * sigmoid(1.702 * x)
 */
__INFINI_C __export infiniStatus_t infiniopCreateQuickGeluDescriptor(
    infiniopHandle_t handle,
    infiniopQuickGeluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x);

/**
 * Query workspace size
 */
__INFINI_C __export infiniStatus_t infiniopGetQuickGeluWorkspaceSize(
    infiniopQuickGeluDescriptor_t desc,
    size_t *size);

/**
 * Launch QuickGELU operator
 */
__INFINI_C __export infiniStatus_t infiniopQuickGelu(
    infiniopQuickGeluDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

/**
 * Destroy descriptor
 */
__INFINI_C __export infiniStatus_t infiniopDestroyQuickGeluDescriptor(
    infiniopQuickGeluDescriptor_t desc);

#endif
