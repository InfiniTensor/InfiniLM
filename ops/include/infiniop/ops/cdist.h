#ifndef __INFINIOP_CDIST_API_H__
#define __INFINIOP_CDIST_API_H__

#include "../operator_descriptor.h"

// 定义 cdist 算子描述符类型
typedef struct InfiniopDescriptor *infiniopCdistDescriptor_t;

/**
 * @brief 创建 Cdist 算子描述符
 * @param handle 算子句柄
 * @param desc_ptr 指向返回的描述符指针
 * @param y 输出张量描述符 (Shape: M x N)
 * @param x1 输入张量1描述符 (Shape: M x D)
 * @param x2 输入张量2描述符 (Shape: N x D)
 * @param p 范数阶数 (L-p norm)
 */
__INFINI_C __export infiniStatus_t infiniopCreateCdistDescriptor(
    infiniopHandle_t handle,
    infiniopCdistDescriptor_t *desc_ptr, // 注意这里应该是具体类型的指针
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    double p);

/**
 * @brief 获取 Cdist 计算所需的临时空间大小
 */
__INFINI_C __export infiniStatus_t infiniopGetCdistWorkspaceSize(infiniopCdistDescriptor_t desc,
                                                                 size_t *size);

/**
 * @brief 执行 Cdist 计算
 * @param desc 算子描述符
 * @param workspace 临时空间指针
 * @param workspace_size 临时空间大小
 * @param y 输出数据指针
 * @param x1 输入1数据指针
 * @param x2 输入2数据指针
 * @param stream 计算流 (CUDA stream 等)
 */
__INFINI_C __export infiniStatus_t infiniopCdist(
    infiniopCdistDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x1,
    const void *x2,
    void *stream);

/**
 * @brief 销毁 Cdist 算子描述符
 */
__INFINI_C __export infiniStatus_t infiniopDestroyCdistDescriptor(infiniopCdistDescriptor_t desc);

#endif
