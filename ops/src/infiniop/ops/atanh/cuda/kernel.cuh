#ifndef __ATANH_CUDA_H__
#define __ATANH_CUDA_H__

namespace op::atanh::cuda {
typedef struct AtanhOp {
public:
    // atanh 是一元算子，只需要一个输入
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, half2>) {
            // 对 half2 的两个部分分别求 atanh
            float2 f = __half22float2(a);
            f.x = atanhf(f.x);
            f.y = atanhf(f.y);
            return __float22half2_rn(f);
        } else if constexpr (std::is_same_v<T, half>) {
            // half 类型先转为 float 计算再转回
            return __float2half(atanhf(__half2float(a)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // bfloat16 类型处理同上
            return __float2bfloat16(atanhf(__bfloat162float(a)));
        } else if constexpr (std::is_same_v<T, float>) {
            // float 直接调用标准数学库函数
            return atanhf(a);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::atanh(a);
        } else {
            // 其他整数类型或不支持类型理论上不应进入，此处做简单 fallback
            return static_cast<T>(atanhf(static_cast<float>(a)));
        }
    }
} AtanhOp;
} // namespace op::atanh::cuda

#endif // __ATANH_CUDA_H__
