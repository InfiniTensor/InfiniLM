#ifndef __INFINIOP_GELUTANH_API_H__
#define __INFINIOP_GELUTANH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGeluTanhDescriptor_t;

/**
 * Create GELU-Tanh descriptor
 *
 * y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
__INFINI_C __export infiniStatus_t infiniopCreateGeluTanhDescriptor(
    infiniopHandle_t handle,
    infiniopGeluTanhDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x);

/**
 * Query workspace size
 */
__INFINI_C __export infiniStatus_t infiniopGetGeluTanhWorkspaceSize(
    infiniopGeluTanhDescriptor_t desc,
    size_t *size);

/**
 * Launch GELU-Tanh operator
 */
__INFINI_C __export infiniStatus_t infiniopGeluTanh(
    infiniopGeluTanhDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

/**
 * Destroy descriptor
 */
__INFINI_C __export infiniStatus_t infiniopDestroyGeluTanhDescriptor(
    infiniopGeluTanhDescriptor_t desc);

#endif
