#ifndef __ADDCMUL_CUDA_CUH__
#define __ADDCMUL_CUDA_CUH__

#include <type_traits>

namespace op::addcmul::cuda {

struct AddcmulOp {
public:
    // addcmul 是三元算子：out = input + value * t1 * t2
    static constexpr size_t num_inputs = 3;

    template <typename T>
    __device__ __host__ __forceinline__ T operator()(const T &input, const T &t1, const T &t2, float value) const {
        float v = value;
        if constexpr (std::is_same_v<T, half>) {
            // 提升至 float 计算以保证精度并简化标量乘法
            float f_input = __half2float(input);
            float f_t1 = __half2float(t1);
            float f_t2 = __half2float(t2);
            return __float2half(f_input + v * f_t1 * f_t2);

        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float f_input = __bfloat162float(input);
            float f_t1 = __bfloat162float(t1);
            float f_t2 = __bfloat162float(t2);
            return __float2bfloat16(f_input + v * f_t1 * f_t2);

        } else if constexpr (std::is_same_v<T, float>) {
            return input + v * t1 * t2;

        } else if constexpr (std::is_same_v<T, double>) {
            return input + static_cast<double>(v) * t1 * t2;

        } else {
            // 兜底逻辑
            return static_cast<T>(static_cast<float>(input) + v * static_cast<float>(t1) * static_cast<float>(t2));
        }
    }
};

} // namespace op::addcmul::cuda

#endif // __ADDCMUL_CUDA_CUH__
