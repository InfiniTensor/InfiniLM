#include "../../../devices/moore/moore_handle.h"
#include "cdist_moore.h"
#include <iostream>
#include <musa_runtime.h>

namespace op::cdist::moore {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    double p) {

    // 1. 转换至 Moore 句柄
    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    // 保持与原版一致，目前仅支持 F32
    CHECK_DTYPE(dtype, INFINI_DTYPE_F32);

    auto result = CdistInfo::create(y_desc, x1_desc, x2_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), p, 0,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// --- Kernel: Generic P-Norm (MUSA F32 实现) ---
__global__ void cdist_generic_kernel_f32(
    float *y,
    const float *x1,
    const float *x2,
    size_t m,
    size_t n,
    size_t d,
    ptrdiff_t x1_stride,
    ptrdiff_t x1_row_stride,
    ptrdiff_t x1_col_stride,
    ptrdiff_t x2_stride,
    ptrdiff_t x2_row_stride,
    ptrdiff_t x2_col_stride,
    ptrdiff_t y_stride,
    ptrdiff_t y_row_stride,
    ptrdiff_t y_col_stride,
    double p) {

    // 2. MUSA 同样支持 3D 线程索引
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (i >= (int)m || j >= (int)n) {
        return;
    }

    // 定位输出 y[b, i, j]
    float *y_ptr = y + b * y_stride + i * y_row_stride + j * y_col_stride;

    // 定位输入向量
    const float *x1_vec = x1 + b * x1_stride + i * x1_row_stride;
    const float *x2_vec = x2 + b * x2_stride + j * x2_row_stride;

    double dist = 0.0;

    for (size_t k = 0; k < d; ++k) {
        float v1 = *(x1_vec + k * x1_col_stride);
        float v2 = *(x2_vec + k * x2_col_stride);
        float diff = fabsf(v1 - v2);

        if (p == 1.0) {
            dist += (double)diff;
        } else if (p == 2.0) {
            dist += (double)diff * diff;
        } else if (isinf(p)) {
            dist = fmaxf((float)dist, diff);
        } else {
            dist += powf(diff, (float)p);
        }
    }

    if (p == 2.0) {
        dist = sqrtf((float)dist);
    } else if (!isinf(p) && p != 1.0) {
        dist = powf((float)dist, 1.0f / (float)p);
    }

    *y_ptr = (float)dist;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x1,
    const void *x2,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    if (_dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // 3. 切换至 musaStream_t
    musaStream_t mustream = reinterpret_cast<musaStream_t>(stream);

    // 保持 16x16 的 Block 大小，这在 MUSA 架构上也是通用的
    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((_info.n + block.x - 1) / block.x),
        static_cast<unsigned int>((_info.m + block.y - 1) / block.y),
        static_cast<unsigned int>(_info.batch));

    cdist_generic_kernel_f32<<<grid, block, 0, mustream>>>(
        static_cast<float *>(y),
        static_cast<const float *>(x1),
        static_cast<const float *>(x2),
        _info.m,
        _info.n,
        _info.d,
        _info.x1_matrix.stride,
        _info.x1_matrix.row_stride,
        _info.x1_matrix.col_stride,
        _info.x2_matrix.stride,
        _info.x2_matrix.row_stride,
        _info.x2_matrix.col_stride,
        _info.y_matrix.stride,
        _info.y_matrix.row_stride,
        _info.y_matrix.col_stride,
        _p);

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::cdist::moore
