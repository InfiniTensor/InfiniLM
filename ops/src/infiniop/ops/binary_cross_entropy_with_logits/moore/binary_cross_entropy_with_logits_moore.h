#ifndef __BINARY_CROSS_ENTROPY_WITH_LOGITS_MOORE_H__
#define __BINARY_CROSS_ENTROPY_WITH_LOGITS_MOORE_H__

#include "../binary_cross_entropy_with_logits.h"

/**
 * 使用 bce_with_logits.h 中定义的 DESCRIPTOR 宏。
 * 这将在命名空间 op::bce_with_logits::moore 中生成针对 Moore 设备的 Descriptor 类。
 * * 在 Moore 端的实现（.mu 文件）中，Opaque 结构体通常包含：
 * - musaHandle_t: 如果使用 MUSA 库的算子实现。
 * - KernelConfig: 用于 MUSA Kernel 的网格（Grid）和线程块（Block）配置。
 * - dataType: 对应的 MUSA 数据类型（如 MUSA_R_32F）。
 */
DESCRIPTOR(moore)

#endif // __BINARY_CROSS_ENTROPY_WITH_LOGITS_MOORE_H__
