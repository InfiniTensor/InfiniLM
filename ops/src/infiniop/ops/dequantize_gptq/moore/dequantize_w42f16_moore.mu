#include "../../../devices/moore/moore_handle.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "dequantize_w42f16_moore.h"

#include "../dequantize_gptq.h"
#include <musa_fp16.h>
#include <cstdint>

// 对齐 nvidia 版：支持 g_idx
// qweight: [in_packed, out_features]，每个 uint32 打包 8 个输入通道的 4-bit
// zeros:   [num_groups, out_packed]，每个 uint32 打包 8 个输出通道的 4-bit
// scales:  [num_groups, out_features]
// g_idx:   [in_features]
__global__ void __launch_bounds__(128)
dequantize_weights_gptq(const uint32_t *__restrict__ qweight,
                        const half     *__restrict__ scales,
                        const uint32_t *__restrict__ zeros,
                        const int      *__restrict__ g_idx,
                        half           *__restrict__ out,
                        int in_features,
                        int out_features,
                        int out_packed,   // ceil(out_features / 8)
                        int num_groups) {
    const int col_pack = blockIdx.x * blockDim.x + threadIdx.x; // packed output column
    const int row      = blockIdx.y * blockDim.y + threadIdx.y; // real input row
    if (col_pack >= out_packed || row >= in_features) return;

    // clamp gid to [0, num_groups)
    const int gid_raw = g_idx ? g_idx[row] : 0;
    const int gid = ((gid_raw % num_groups) + num_groups) % num_groups;

    const int pack_row = row >> 3;          // packed input row (8 rows per pack)
    const int q_shift  = (row & 7) * 4;     // nibble shift within uint32

    const int zero_idx = gid * out_packed + col_pack;
    const uint32_t zeros_loaded = zeros[zero_idx];

    const int col_base   = col_pack << 3;  // 8 real cols per pack
    const int scale_base = gid * out_features + col_base;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = col_base + j;
        if (col >= out_features) break;

        const uint32_t q_loaded = qweight[pack_row * out_features + col];
        const int q_nib = (q_loaded >> q_shift) & 0xF;

        const int z_nib = (zeros_loaded >> (j * 4)) & 0xF;
        const half scale = scales[scale_base + j];

        // 与 nvidia 版一致： (q - (z + 1)) * s
        const float v = float(q_nib - (z_nib + 1)) * __half2float(scale);
        out[row * out_features + col] = __float2half(v);
    }
}

namespace op::dequantize_gptq::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
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
    infiniopTensorDescriptor_t zeros_desc,
    infiniopTensorDescriptor_t g_idx_desc) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto result = DequantizeGPTQInfo::create(out_desc, qweight_desc, scales_desc, zeros_desc, g_idx_desc);

    *desc_ptr = new Descriptor(
        0,
        new Opaque{handle->internal()},
        result.take(),
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *qweight,
    const void *scales,
    const void *zeros,
    const void *g_idx,
    void *stream) const {

    const int in_features  = _info.in_features();
    const int out_features = _info.out_features();
    const int out_packed   = _info.out_packed();
    const int in_packed    = _info.in_packed();
    const int num_groups   = _info.num_groups();

    if (num_groups <= 0 || in_features <= 0 || out_features <= 0 || out_packed <= 0 || in_packed <= 0)
        return INFINI_STATUS_BAD_PARAM;

    constexpr int BLOCK_X = 16; // packed columns
    constexpr int BLOCK_Y = 4;  // rows
    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks((out_packed + BLOCK_X - 1) / BLOCK_X,
                (in_features + BLOCK_Y - 1) / BLOCK_Y);

    dequantize_weights_gptq<<<blocks, threads, 0, reinterpret_cast<musaStream_t>(stream)>>>(
        reinterpret_cast<const uint32_t*>(qweight),
        reinterpret_cast<const half*>(scales),
        reinterpret_cast<const uint32_t*>(zeros),
        reinterpret_cast<const int*>(g_idx),
        reinterpret_cast<half*>(out),
        in_features, out_features, out_packed, num_groups);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dequantize_gptq::moore
