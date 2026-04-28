#ifndef __SOFTPLUS_KUNLUN_KERNEL_H__
#define __SOFTPLUS_KUNLUN_KERNEL_H__

namespace op::softplus::kunlun {

typedef struct SoftplusOp {
public:
    static constexpr int num_inputs = 1;
    template <typename T>
    inline __device__ T operator()(const T *inputs, float beta, float threshold) const {
        if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(inputs[0]) * beta;
            float out = (xf > threshold) ? xf : log(1 + exp(xf)) / beta;
            return __float2half(out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            float xf = __bfloat162float(inputs[0]) * beta;
            float out = (xf > threshold) ? xf : log(1 + exp(xf)) / beta;
            return __float2bfloat16(out);
        } else {
            float xf = inputs[0] * beta;
            return (xf > threshold) ? xf : log(1 + exp(xf)) / beta;
        }
    }
} SoftplusOp;

} // namespace op::softplus::kunlun

#endif // __SOFTPLUS_KUNLUN_KERNEL_H__
