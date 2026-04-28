#include "affine_grid_moore.h"
#include "affine_grid_moore_kernel.h"
#include <musa_runtime.h>

// 引用 Handle 路径
#include "../../../devices/moore/moore_handle.h"

namespace op::affine_grid::moore {

template <typename T>
__global__ void affine_grid_kernel(
    const int N, const int H, const int W,
    const bool align_corners,
    const T *theta,
    T *output) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * H * W;

    if (idx < total_pixels) {
        int w = idx % W;
        int h = (idx / W) % H;
        int n = idx / (W * H);

        const T *current_theta = theta + n * 6;
        T *out_ptr = output + idx * 2;

        AffineGridOp op;
        op(w, h, W, H, current_theta, align_corners, &out_ptr[0], &out_ptr[1]);
    }
}

// ==================================================================
// 2. Launcher Implementation
// ==================================================================

template <typename T>
void affine_grid_moore_launch(
    const AffineGridInfo &info,
    T *output,
    const T *input,
    void *stream) {

    size_t num_pixels = info.batch() * info.height() * info.width();

    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;

    affine_grid_kernel<T><<<blocks, threads, 0, (musaStream_t)stream>>>(
        info.batch(),
        info.height(),
        info.width(),
        info.align_corners(),
        input,
        output);
}

// ==================================================================
// 3. Descriptor Implementation
// ==================================================================

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    bool align_corners) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto info_result = AffineGridInfo::create(out_desc, in_desc, align_corners);

    if (!info_result) {

        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(
        nullptr,
        *info_result,
        0,
        handle->device,   // 原: handle->device_type()
        handle->device_id // 原: handle->device_id()
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        affine_grid_moore_launch<half>(
            _info,
            static_cast<half *>(output),
            static_cast<const half *>(input),
            stream);
        break;

    case INFINI_DTYPE_BF16:

        affine_grid_moore_launch<__mt_bfloat16>(
            _info,
            static_cast<__mt_bfloat16 *>(output),
            static_cast<const __mt_bfloat16 *>(input),
            stream);
        break;

    case INFINI_DTYPE_F32:
        affine_grid_moore_launch<float>(
            _info,
            static_cast<float *>(output),
            static_cast<const float *>(input),
            stream);
        break;

    case INFINI_DTYPE_F64:
        affine_grid_moore_launch<double>(
            _info,
            static_cast<double *>(output),
            static_cast<const double *>(input),
            stream);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::affine_grid::moore
