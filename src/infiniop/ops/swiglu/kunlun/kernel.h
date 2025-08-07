#ifndef __SWIGLU_KUNLUN_KERNEL_H__
#define __SWIGLU_KUNLUN_KERNEL_H__

namespace op::swiglu::kunlun {

/// @brief SwiGLU op kernel
typedef struct SwiGLUOp {
private:
    template <typename T>
    inline __device__ T sigmoid(T x) const {
        return 1.0f / (1.0f + exp(-x));
    }
    // float version of sigmoid
    inline __device__ float sigmoidf(float x) const {
        return 1.0f / (1.0f + exp(-x));
    }

public:
    // This static number must be set in other Ops
    static constexpr int num_inputs = 2;
    template <typename T>
    inline __device__ T operator()(const T *inputs) const {
        T up = inputs[0];
        T gate = inputs[1];
        T out = gate * sigmoid(gate) * up;
        return out;
    }
    // bfloat16 特化版本（使用 float 计算精度）
    inline __device__ bfloat16_t operator()(const bfloat16_t *inputs) const {
        float up_f = __bfloat162float(inputs[0]);
        float gate_f = __bfloat162float(inputs[1]);

        float out_f = gate_f * sigmoidf(gate_f) * up_f;
        return __float2bfloat16(out_f);
    }
} SwiGLUOp;
} // namespace op::swiglu::kunlun

#endif // __SWIGLU_KUNLUN_KERNEL_H__
