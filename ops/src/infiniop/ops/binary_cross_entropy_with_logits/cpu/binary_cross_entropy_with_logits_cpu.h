#ifndef __BINARY_CROSS_ENTROPY_WITH_LOGITS_CPU_H__
#define __BINARY_CROSS_ENTROPY_WITH_LOGITS_CPU_H__

#include "../binary_cross_entropy_with_logits.h"

/**
 * 使用 bce_with_logits.h 中定义的 DESCRIPTOR 宏
 * * 这将自动在命名空间 op::bce_with_logits::cpu 中生成 Descriptor 类。
 * 该类将继承自 InfiniopDescriptor，并包含：
 * - BCEWithLogitsInfo _info (存储校验后的维度和步长)
 * - create() 静态方法 (负责 CPU 版描述符的实例化)
 * - calculate() 方法 (负责 CPU 版数值稳定逻辑的执行)
 */
DESCRIPTOR(cpu)

#endif // __BINARY_CROSS_ENTROPY_WITH_LOGITS_CPU_H__
