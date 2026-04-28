#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API)

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dequantize_w42f16_kernel.cuh"
#include "dequantize_w42f16_nvidia.cuh"

#include "../dequantize_awq.h"
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
__global__ void __launch_bounds__(64)
    dequantize_weights_awq(int *__restrict__ B, half *__restrict__ scaling_factors,
                           int *__restrict__ zeros, half *__restrict__ C, int G,
                           int out_features, int in_features) {
    // static constexpr uint32_t ZERO = 0x0;

    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    int row = (blockIdx.y * blockDim.y + threadIdx.y);

    // 边界检查，防止越界访问
    if (col >= out_features || row >= in_features) {
        return;
    }

    // 每个元素在输出中的起始地址：行主序，连续 8 个 half
    int index1 = 8 * col + 8 * row * out_features;
    half *C_ptr2 = C + index1;

    int index2 = col + row * out_features;
    int *B_ptr2 = B + index2;

    int index3 = col + (int)(row / G) * out_features;
    int *zeros_ptr2 = zeros + index3;

    int index4 = 8 * col + (int)(row / G) * out_features * 8;
    half *scaling_factors_ptr2 = scaling_factors + index4;

    uint32_t zeros_loaded = *(uint32_t *)(zeros_ptr2);
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2_awq(zeros_loaded);
    uint4 B_loaded_scale = *(uint4 *)(scaling_factors_ptr2);

    uint32_t B_loaded = *(uint32_t *)B_ptr2;
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_awq(B_loaded);

    // Reinterpret uint4 components as __half2
    __half2 *B_loaded_fp16_h2 = reinterpret_cast<__half2 *>(&B_loaded_fp16);
    __half2 *B_loaded_zero_h2 = reinterpret_cast<__half2 *>(&B_loaded_zero);
    __half2 *B_loaded_scale_h2 = reinterpret_cast<__half2 *>(&B_loaded_scale);

    // Replace PTX sub.f16x2 with __hsub2 for each component
    B_loaded_fp16_h2[0] = __hsub2(B_loaded_fp16_h2[0], B_loaded_zero_h2[0]);
    B_loaded_fp16_h2[1] = __hsub2(B_loaded_fp16_h2[1], B_loaded_zero_h2[1]);
    B_loaded_fp16_h2[2] = __hsub2(B_loaded_fp16_h2[2], B_loaded_zero_h2[2]);
    B_loaded_fp16_h2[3] = __hsub2(B_loaded_fp16_h2[3], B_loaded_zero_h2[3]);

    // Replace PTX fma.rn.f16x2 with __hfma2 for each component
    B_loaded_fp16_h2[0] = __hfma2(B_loaded_fp16_h2[0], B_loaded_scale_h2[0], __float2half2_rn(0.0f));
    B_loaded_fp16_h2[1] = __hfma2(B_loaded_fp16_h2[1], B_loaded_scale_h2[1], __float2half2_rn(0.0f));
    B_loaded_fp16_h2[2] = __hfma2(B_loaded_fp16_h2[2], B_loaded_scale_h2[2], __float2half2_rn(0.0f));
    B_loaded_fp16_h2[3] = __hfma2(B_loaded_fp16_h2[3], B_loaded_scale_h2[3], __float2half2_rn(0.0f));

    // 直接写回全局内存输出
    half *out_vec = reinterpret_cast<half *>(&B_loaded_fp16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        C_ptr2[i] = out_vec[i];
    }
}
#else
__global__ void __launch_bounds__(64)
    dequantize_weights_awq(int *__restrict__ B, half *__restrict__ scaling_factors,
                           int *__restrict__ zeros, half *__restrict__ C, int group_size,
                           int out_features, int in_features) {
    static constexpr uint32_t ZERO = 0x0;

    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，防止越界访问
    if (col >= out_features || row >= in_features) {
        return;
    }

    int index1 = 8 * col + 8 * row * out_features;
    half *C_ptr2 = C + index1;

    int index2 = col + row * out_features;
    int *B_ptr2 = B + index2;

    int index3 = col + (int)(row / group_size) * out_features;
    int *zeros_ptr2 = zeros + index3;
    int index4 = 8 * col + (int)(row / group_size) * out_features * 8;
    half *scaling_factors_ptr2 = scaling_factors + index4;

    uint32_t zeros_loaded = *(uint32_t *)(zeros_ptr2);
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2_awq(zeros_loaded);
    uint4 B_loaded_scale = *(uint4 *)(scaling_factors_ptr2);

    uint32_t B_loaded = *(uint32_t *)B_ptr2;
    uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2_awq(B_loaded);
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(B_loaded_fp16.x)
                 : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(B_loaded_fp16.x)
                 : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(B_loaded_fp16.y)
                 : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(B_loaded_fp16.y)
                 : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(B_loaded_fp16.z)
                 : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(B_loaded_fp16.z)
                 : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(B_loaded_fp16.w)
                 : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(B_loaded_fp16.w)
                 : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

    // 直接写回全局内存输出
    half *out_vec = reinterpret_cast<half *>(&B_loaded_fp16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        C_ptr2[i] = out_vec[i];
    }
}
#endif

namespace op::dequantize_awq::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t scales_desc,
    infiniopTensorDescriptor_t zeros_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto result = DequantizeAWQInfo::create(out_desc, qweight_desc, scales_desc, zeros_desc);

    *desc_ptr = new Descriptor(
        0,
        new Opaque{handle->internal()},
        result.take(),
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t
Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *qweight,
    const void *scales,
    const void *zeros,
    void *stream) const {
    int in_features = _info.in_features();
    int out_features = _info.out_features();
    int group_size = in_features / _info.num_groups();

    // ==================== 默认配置, 固定为 8 ====================
    constexpr int BLOCK_X = 8;
    constexpr int BLOCK_Y = 8;

    int x_blocks = (out_features + BLOCK_X - 1) / BLOCK_X;
    int y_blocks = (in_features + BLOCK_Y - 1) / BLOCK_Y;

    dim3 num_blocks(x_blocks, y_blocks);
    dim3 threads_per_block(BLOCK_X, BLOCK_Y);
    // =====================================================

    half *out_ = reinterpret_cast<half *>(out);

    int *qweight_ = const_cast<int *>(reinterpret_cast<const int *>(qweight));
    half *scales_ = const_cast<half *>(reinterpret_cast<const half *>(scales));
    int *zeros_ = const_cast<int *>(reinterpret_cast<const int *>(zeros));

    dequantize_weights_awq<<<num_blocks, threads_per_block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        qweight_, scales_, zeros_, out_, group_size, out_features, in_features);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dequantize_awq::nvidia

#endif
