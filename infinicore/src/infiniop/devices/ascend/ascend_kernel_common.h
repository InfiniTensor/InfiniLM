#ifndef __INFINIOP_ASCEND_KERNEL_COMMON_H__
#define __INFINIOP_ASCEND_KERNEL_COMMON_H__

#include "../../../../include/infinicore.h"
#include "kernel_operator.h"

constexpr size_t BLOCK_NUM = 8;
constexpr size_t BUFFER_NUM = 2;
constexpr size_t BYTE_ALIGN = 32;
constexpr size_t BLOCK_LEN = 256;

template <typename T>
__aicore__ inline size_t alignTileLen(size_t tile_len, size_t byte_align) {
    size_t bytes = tile_len * sizeof(T);
    size_t aligned_bytes = (bytes % byte_align == 0)
                             ? bytes
                             : (bytes + (byte_align - bytes % byte_align));
    return aligned_bytes / sizeof(T);
}

#endif
