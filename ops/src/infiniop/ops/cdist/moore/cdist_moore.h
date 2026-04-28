#ifndef __CDIST_MOORE_H__
#define __CDIST_MOORE_H__

#include "../cdist.h"

/**
 * 使用 cdist.h 中定义的 DESCRIPTOR 宏。
 * 这将在命名空间 op::cdist::moore 中生成针对 Moore 设备的 Descriptor 类。
 * * 在 Moore 端的具体实现中，Opaque 结构体通常会存储：
 * - mublasHandle_t: 用于 p=2.0 时的矩阵乘法加速（对应 NVIDIA 的 cuBLAS）。
 * - musaStream_t: 当前执行的任务流。
 * - 自定义 Kernel 的配置参数。
 */
DESCRIPTOR(moore)

#endif // __CDIST_MOORE_H__
