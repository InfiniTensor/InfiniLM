#include "addbmm_moore.h"

#include "../../../devices/moore/moore_handle.h"
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <vector>

namespace op::addbmm::moore {

// ==================================================================
// 1. Kernel Implementation (Removed Macros)
// ==================================================================

template <typename T>
__global__ void addbmm_kernel(
    const size_t B, const size_t N, const size_t M, const size_t P,
    const float alpha, const float beta,
    T *output,
    const T *input,
    const T *batch1,
    const T *batch2,
    const int64_t out_s0, const int64_t out_s1,
    const int64_t in_s0, const int64_t in_s1,
    const int64_t b1_s0, const int64_t b1_s1, const int64_t b1_s2,
    const int64_t b2_s0, const int64_t b2_s1, const int64_t b2_s2) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * P;

    if (idx < total_elements) {
        size_t n = idx / P;
        size_t p = idx % P;

        float matmul_sum = 0.0f;

        // 预先计算与 b 无关的偏移量部分，略微优化性能
        int64_t b1_n_offset = n * b1_s1;
        int64_t b2_p_offset = p * b2_s2;

        for (size_t b = 0; b < B; ++b) {
            int64_t b1_b_offset = b * b1_s0;
            int64_t b2_b_offset = b * b2_s0;

            for (size_t m = 0; m < M; ++m) {
                // 直接计算偏移：Batch1[b, n, m]
                int64_t offset1 = b1_b_offset + b1_n_offset + m * b1_s2;
                // 直接计算偏移：Batch2[b, m, p]
                int64_t offset2 = b2_b_offset + m * b2_s1 + b2_p_offset;

                T val1 = batch1[offset1];
                T val2 = batch2[offset2];

                float v1_f, v2_f;
                if constexpr (std::is_same_v<T, half>) {
                    v1_f = __half2float(val1);
                    v2_f = __half2float(val2);
                } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                    v1_f = __bfloat162float(val1);
                    v2_f = __bfloat162float(val2);
                } else {
                    v1_f = static_cast<float>(val1);
                    v2_f = static_cast<float>(val2);
                }
                matmul_sum += v1_f * v2_f;
            }
        }

        // 直接计算偏移：Input[n, p]
        int64_t in_offset = n * in_s0 + p * in_s1;
        T in_val = input[in_offset];

        float in_val_f;
        if constexpr (std::is_same_v<T, half>) {
            in_val_f = __half2float(in_val);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            in_val_f = __bfloat162float(in_val);
        } else {
            in_val_f = static_cast<float>(in_val);
        }

        float result = beta * in_val_f + alpha * matmul_sum;

        // 直接计算偏移：Output[n, p]
        int64_t out_offset = n * out_s0 + p * out_s1;

        if constexpr (std::is_same_v<T, half>) {
            output[out_offset] = __float2half(result);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            output[out_offset] = __float2bfloat16(result);
        } else {
            output[out_offset] = static_cast<T>(result);
        }
    }
}

// ==================================================================
// 2. Launcher Implementation
// ==================================================================

template <typename T>
void addbmm_moore_launch(
    const AddbmmInfo &info,
    T *output,
    const T *input,
    const T *batch1,
    const T *batch2,
    void *stream) {

    size_t total_elements = info.n() * info.p();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    const auto &out_strides = info.out_strides();
    const auto &in_strides = info.in_strides();
    const auto &b1_strides = info.b1_strides();
    const auto &b2_strides = info.b2_strides();

    addbmm_kernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(
        info.b(), info.n(), info.m(), info.p(),
        info.alpha(), info.beta(),
        output, input, batch1, batch2,
        out_strides[0], out_strides[1],
        in_strides[0], in_strides[1],
        b1_strides[0], b1_strides[1], b1_strides[2],
        b2_strides[0], b2_strides[1], b2_strides[2]);
}

// ==================================================================
// 3. Descriptor Implementation
// ==================================================================

Descriptor::~Descriptor() = default;

// 匹配 std::vector 接口
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float alpha,
    float beta) {

    if (input_desc_vec.size() != 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    infiniopTensorDescriptor_t in_desc = input_desc_vec[0];
    infiniopTensorDescriptor_t batch1_desc = input_desc_vec[1];
    infiniopTensorDescriptor_t batch2_desc = input_desc_vec[2];

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto info_result = AddbmmInfo::create(out_desc, in_desc, batch1_desc, batch2_desc, alpha, beta);

    if (!info_result) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(
        nullptr,
        *info_result,
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// 匹配 std::vector 接口
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.size() != 3) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const void *input = inputs[0];
    const void *batch1 = inputs[1];
    const void *batch2 = inputs[2];

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        addbmm_moore_launch<half>(
            _info,
            static_cast<half *>(output),
            static_cast<const half *>(input),
            static_cast<const half *>(batch1),
            static_cast<const half *>(batch2),
            stream);
        break;

    case INFINI_DTYPE_BF16:
        addbmm_moore_launch<__mt_bfloat16>(
            _info,
            static_cast<__mt_bfloat16 *>(output),
            static_cast<const __mt_bfloat16 *>(input),
            static_cast<const __mt_bfloat16 *>(batch1),
            static_cast<const __mt_bfloat16 *>(batch2),
            stream);
        break;

    case INFINI_DTYPE_F32:
        addbmm_moore_launch<float>(
            _info,
            static_cast<float *>(output),
            static_cast<const float *>(input),
            static_cast<const float *>(batch1),
            static_cast<const float *>(batch2),
            stream);
        break;

    case INFINI_DTYPE_F64:
        addbmm_moore_launch<double>(
            _info,
            static_cast<double *>(output),
            static_cast<const double *>(input),
            static_cast<const double *>(batch1),
            static_cast<const double *>(batch2),
            stream);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::addbmm::moore
