#ifndef __ZEROS_MOORE_KERNEL_H__
#define __ZEROS_MOORE_KERNEL_H__

#include <cuda_fp8.h>
namespace op::zeros::cuda {
typedef struct ZerosOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, bool>) { // 1
            return false;
        } else if constexpr (std::is_same_v<T, uint8_t>) { // 2
            return 0;
        } else if constexpr (std::is_same_v<T, int8_t>) { // 3
            return 0;
        } else if constexpr (std::is_same_v<T, int16_t>) { // 4
            return 0;
        } else if constexpr (std::is_same_v<T, int32_t>) { // 5
            return 0;
        } else if constexpr (std::is_same_v<T, int64_t>) { // 6
            return 0;
        } else if constexpr (std::is_same_v<T, uint8_t>) { // 7
            return 0;
        } else if constexpr (std::is_same_v<T, uint16_t>) { // 8
            return 0;
        } else if constexpr (std::is_same_v<T, uint32_t>) { // 9
            return 0;
        } else if constexpr (std::is_same_v<T, uint64_t>) { // 10
            return 0;
        } else if constexpr (std::is_same_v<T, cuda_fp8_e4m3>) { // 11
            return cuda_fp8_e4m3(0.0f);
        } else if constexpr (std::is_same_v<T, half>) { // 12
            return __float2half(0.0f);
        } else if constexpr (std::is_same_v<T, float>) { // 13
            return 0.0f;
        } else if constexpr (std::is_same_v<T, double>) { // 14
            return 0.0;
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) { // 19
            return __float2bfloat16(0.0f);
        } else {
            return 0.0;
        }
    }
} ZerosOp;
} // namespace op::zeros::cuda

#endif // __ZEROS_MOORE_KERNEL_H__
