#ifndef __GELU_KUNLUN_KERNEL_H__
#define __GELU_KUNLUN_KERNEL_H__

namespace op::gelu::kunlun {

typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    inline __device__ T operator()(const T *x) const {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            float x_f = __bfloat162float(x[0]);
            float result = 0.5 * x_f * (1 + fast_erf(x_f / sqrt(2.0f)));

            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x[0]);
            float result = 0.5 * x_f * (1 + fast_erf(x_f / sqrt(2.0f)));

            return __float2half(result);
        } else if constexpr (std::is_same_v<T, float>) {

            return 0.5 * x[0] * (1 + fast_erf(x[0] / sqrt(2.0f)));
        } else {
            return 0.5 * x[0] * (1 + fast_erf(x[0] / sqrt(2.0)));
        }
    }
} GeluOp;

} // namespace op::gelu::kunlun

#endif // __GELU_KUNLUN_H__
