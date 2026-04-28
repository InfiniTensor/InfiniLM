#ifndef __CDIST_NVIDIA_CUH__
#define __CDIST_NVIDIA_CUH__

#include "../cdist.h"

/**
 * 使用 cdist.h 中定义的 DESCRIPTOR 宏。
 * 这将在命名空间 op::cdist::nvidia 中生成针对 NVIDIA 设备的 Descriptor 类。
 * * 在 NVIDIA 端的具体实现中，Opaque 结构体通常会存储：
 * - cublasHandle_t: 用于 p=2.0 时的矩阵乘法加速。
 * - cudaStream_t: 当前执行的任务流。
 * - 自定义 Kernel 的配置参数。
 */
DESCRIPTOR(nvidia)

#endif // __CDIST_NVIDIA_CUH__
