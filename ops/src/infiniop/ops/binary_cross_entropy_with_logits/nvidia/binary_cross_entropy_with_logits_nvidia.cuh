#ifndef __BINARY_CROSS_ENTROPY_WITH_LOGITS_NVIDIA_CUH__
#define __BINARY_CROSS_ENTROPY_WITH_LOGITS_NVIDIA_CUH__

#include "../binary_cross_entropy_with_logits.h"

/**
 * 使用 bce_with_logits.h 中定义的 DESCRIPTOR 宏。
 * 这将在命名空间 op::bce_with_logits::nvidia 中生成针对 NVIDIA 设备的 Descriptor 类。
 * * * 在 NVIDIA 端的实现（.cu 文件）中，Opaque 结构体通常包含：
 * - cudnnHandle_t: 如果使用 cuDNN 的算子实现。
 * - cudnnTensorDescriptor_t: 用于描述各输入输出张量的 cuDNN 格式。
 * - KernelConfig: 用于自定义 CUDA Kernel 的网格（Grid）和线程块（Block）配置。
 * - dataType: 对应的 CUDA 数据类型（如 CUDA_R_32F）。
 */
DESCRIPTOR(nvidia)

#endif // __BINARY_CROSS_ENTROPY_WITH_LOGITS_NVIDIA_CUH__
