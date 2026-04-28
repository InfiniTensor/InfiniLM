#ifndef __INFINIOP_BINARY_CROSS_ENTROPY_WITH_LOGITS_API_H__
#define __INFINIOP_BINARY_CROSS_ENTROPY_WITH_LOGITS_API_H__

#include "../operator_descriptor.h"

// 定义归约方式枚举
typedef enum {
    INFINIOP_REDUCTION_NONE = 0,
    INFINIOP_REDUCTION_MEAN = 1,
    INFINIOP_REDUCTION_SUM = 2
} infiniopReduction_t;

// 定义 BCEWithLogits 算子描述符类型
typedef struct InfiniopDescriptor *infiniopBCEWithLogitsDescriptor_t;

/**
 * @brief 创建 BCEWithLogits 算子描述符
 * @param handle 算子句柄
 * @param desc_ptr 指向返回的描述符指针
 * @param out 输出张量描述符 (none时与input同形状，mean/sum时为标量)
 * @param logits 输入 Logits 张量描述符
 * @param target 目标标签张量描述符
 * @param weight 样本权重描述符 (可选，不需要则传 NULL)
 * @param pos_weight 正样本权重描述符 (可选，不需要则传 NULL)
 * @param reduction 归约方式 (none, mean, sum)
 */
__INFINI_C __export infiniStatus_t infiniopCreateBCEWithLogitsDescriptor(
    infiniopHandle_t handle,
    infiniopBCEWithLogitsDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out,
    infiniopTensorDescriptor_t logits,
    infiniopTensorDescriptor_t target,
    infiniopTensorDescriptor_t weight,
    infiniopTensorDescriptor_t pos_weight,
    infiniopReduction_t reduction);

/**
 * @brief 获取 BCEWithLogits 计算所需的临时空间大小
 */
__INFINI_C __export infiniStatus_t infiniopGetBCEWithLogitsWorkspaceSize(
    infiniopBCEWithLogitsDescriptor_t desc,
    size_t *size);

/**
 * @brief 执行 BCEWithLogits 计算
 * @param desc 算子描述符
 * @param workspace 临时空间指针
 * @param workspace_size 临时空间大小
 * @param out 输出数据指针
 * @param logits Logits 数据指针
 * @param target Target 数据指针
 * @param weight 权重数据指针 (可选，传 NULL 表示权重全为 1)
 * @param pos_weight 正样本权重数据指针 (可选，传 NULL 表示权重全为 1)
 * @param stream 计算流
 */
__INFINI_C __export infiniStatus_t infiniopBCEWithLogits(
    infiniopBCEWithLogitsDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *logits,
    const void *target,
    const void *weight,
    const void *pos_weight,
    void *stream);

/**
 * @brief 销毁 BCEWithLogits 算子描述符
 */
__INFINI_C __export infiniStatus_t infiniopDestroyBCEWithLogitsDescriptor(
    infiniopBCEWithLogitsDescriptor_t desc);

#endif
