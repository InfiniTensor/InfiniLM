#ifndef __INFINIOP_ADDCMUL_API_H__
#define __INFINIOP_ADDCMUL_API_H__

#include "../operator_descriptor.h"

// 定义 addcmul 算子描述符类型
typedef struct InfiniopDescriptor *infiniopAddcmulDescriptor_t;

/**
 * @brief 创建 Addcmul 算子描述符
 * @param handle 算子句柄
 * @param desc_ptr 指向返回的描述符指针
 * @param out 输出张量描述符
 * @param input 加项张量描述符
 * @param tensor1 乘项张量1描述符
 * @param tensor2 乘项张量2描述符
 * @param value 乘积的标量系数
 */
__INFINI_C __export infiniStatus_t infiniopCreateAddcmulDescriptor(infiniopHandle_t handle,
                                                                   infiniopAddcmulDescriptor_t *desc_ptr,
                                                                   infiniopTensorDescriptor_t out,
                                                                   infiniopTensorDescriptor_t input,
                                                                   infiniopTensorDescriptor_t tensor1,
                                                                   infiniopTensorDescriptor_t tensor2,
                                                                   float value);

/**
 * @brief 获取 Addcmul 计算所需的临时空间大小
 */
__INFINI_C __export infiniStatus_t infiniopGetAddcmulWorkspaceSize(infiniopAddcmulDescriptor_t desc, size_t *size);

/**
 * @brief 执行 Addcmul 计算
 * @param desc 算子描述符
 * @param workspace 临时空间指针
 * @param workspace_size 临时空间大小
 * @param out 输出数据指针
 * @param input 加项数据指针
 * @param tensor1 乘项1数据指针
 * @param tensor2 乘项2数据指针
 * @param stream 计算流 (CUDA stream 等)
 */
__INFINI_C __export infiniStatus_t infiniopAddcmul(infiniopAddcmulDescriptor_t desc,
                                                   void *workspace,
                                                   size_t workspace_size,
                                                   void *out,
                                                   const void *input,
                                                   const void *tensor1,
                                                   const void *tensor2,
                                                   void *stream);

/**
 * @brief 销毁 Addcmul 算子描述符
 */
__INFINI_C __export infiniStatus_t infiniopDestroyAddcmulDescriptor(infiniopAddcmulDescriptor_t desc);

#endif
