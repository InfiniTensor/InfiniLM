#ifndef __ADAPTIVE_AVG_POOL1D_MOORE_KERNEL_H__
#define __ADAPTIVE_AVG_POOL1D_MOORE_KERNEL_H__

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <type_traits>

namespace op::adaptive_avg_pool1d::moore {

typedef struct AdaptiveAvgPool1dOp {
public:
    template <typename T>
    __device__ __forceinline__ void operator()(
        const int w_out,
        const int input_size,
        const int output_size,
        const T *input_base,
        T *output_ptr) const {

        int start = (w_out * input_size) / output_size;
        int end = ((w_out + 1) * input_size + output_size - 1) / output_size;

        start = (start < 0) ? 0 : start;
        end = (end > input_size) ? input_size : end;

        int kernel_size = end - start;
        kernel_size = (kernel_size < 1) ? 1 : kernel_size;

        float sum = 0.0f;

        for (int i = start; i < end; ++i) {
            T val = input_base[i];

            if constexpr (std::is_same_v<T, half>) {
                sum += __half2float(val);
            } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                sum += __bfloat162float(val);
            } else {
                sum += static_cast<float>(val);
            }
        }

        float avg = sum / static_cast<float>(kernel_size);

        if constexpr (std::is_same_v<T, half>) {
            *output_ptr = __float2half(avg);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            *output_ptr = __float2bfloat16(avg);
        } else {
            *output_ptr = static_cast<T>(avg);
        }
    }

} AdaptiveAvgPool1dOp;

} // namespace op::adaptive_avg_pool1d::moore

#endif // __ADAPTIVE_AVG_POOL1D_MOORE_KERNEL_H__
