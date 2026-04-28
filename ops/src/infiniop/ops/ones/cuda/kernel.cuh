#ifndef __ONES_CUDA_H__
#define __ONES_CUDA_H__

namespace op::ones::cuda {
typedef struct OnesOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, bool>) { // 1
            return true;
        } else if constexpr (std::is_same_v<T, uint8_t>) { // 2
            return 1;
        } else if constexpr (std::is_same_v<T, int8_t>) { // 3
            return 1;
        } else if constexpr (std::is_same_v<T, int16_t>) { // 4
            return 1;
        } else if constexpr (std::is_same_v<T, int32_t>) { // 5
            return 1;
        } else if constexpr (std::is_same_v<T, int64_t>) { // 6
            return 1;
        } else if constexpr (std::is_same_v<T, uint8_t>) { // 7
            return 1;
        } else if constexpr (std::is_same_v<T, uint16_t>) { // 8
            return 1;
        } else if constexpr (std::is_same_v<T, uint32_t>) { // 9
            return 1;
        } else if constexpr (std::is_same_v<T, uint64_t>) { // 10
            return 1;
#ifndef ENABLE_HYGON_API
        } else if constexpr (std::is_same_v<T, cuda_fp8_e4m3>) { // 11
            return cuda_fp8_e4m3(1.0f);
#endif
        } else if constexpr (std::is_same_v<T, half>) { // 12
            return __float2half(1.0f);
        } else if constexpr (std::is_same_v<T, float>) { // 13
            return 1.0f;
        } else if constexpr (std::is_same_v<T, double>) { // 14
            return 1.0;
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) { // 19
            return __float2bfloat16(1.0f);
        } else {
            return 1.0;
        }
    }
} OnesOp;
} // namespace op::ones::cuda

#endif // __ONES_CUDA_H__
