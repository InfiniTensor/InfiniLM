#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.h"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "lerp_nvidia.cuh"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace op::lerp::nvidia {

// ==================================================================
// 1. 定义公共结构体 (解决 Opaque private 访问权限问题)
// ==================================================================
struct LerpOpaqueData {
    int ndim;

    // Device Pointers
    int64_t *d_shape = nullptr;
    int64_t *d_start_strides = nullptr;
    int64_t *d_end_strides = nullptr;
    int64_t *d_weight_strides = nullptr;
};

// 让 Opaque 继承自 LerpOpaqueData
struct Descriptor::Opaque : public LerpOpaqueData {};

Descriptor::~Descriptor() {
    if (_opaque) {
        if (_opaque->d_shape) {
            cudaFree(_opaque->d_shape);
        }
        if (_opaque->d_start_strides) {
            cudaFree(_opaque->d_start_strides);
        }
        if (_opaque->d_end_strides) {
            cudaFree(_opaque->d_end_strides);
        }
        if (_opaque->d_weight_strides) {
            cudaFree(_opaque->d_weight_strides);
        }
        delete _opaque;
        _opaque = nullptr;
    }
}

// ==================================================================
// 2. 辅助函数
// ==================================================================

static std::vector<int64_t> compute_broadcast_strides(
    const std::vector<size_t> &out_shape,
    infiniopTensorDescriptor_t input_desc) {

    int out_ndim = static_cast<int>(out_shape.size());
    int in_ndim = static_cast<int>(input_desc->ndim());

    // 使用引用接收 vector
    const auto &in_shape = input_desc->shape();
    const auto &in_strides = input_desc->strides();

    std::vector<int64_t> effective_strides(out_ndim, 0);

    for (int i = 0; i < out_ndim; ++i) {
        int out_idx = out_ndim - 1 - i;
        int in_idx = in_ndim - 1 - i;

        if (in_idx >= 0) {
            size_t dim_size = in_shape[in_idx];
            if (dim_size == 1) {
                effective_strides[out_idx] = 0;
            } else {
                effective_strides[out_idx] = in_strides[in_idx];
            }
        } else {
            effective_strides[out_idx] = 0;
        }
    }
    return effective_strides;
}

template <typename T>
static T *upload_to_device(const std::vector<T> &host_vec) {
    if (host_vec.empty()) {
        return nullptr;
    }
    T *d_ptr = nullptr;
    size_t size_bytes = host_vec.size() * sizeof(T);
    cudaMalloc(&d_ptr, size_bytes);
    cudaMemcpy(d_ptr, host_vec.data(), size_bytes, cudaMemcpyHostToDevice);
    return d_ptr;
}

// ==================================================================
// 3. Kernel Launch Logic
// ==================================================================

// 使用 const LerpOpaqueData* 作为参数类型
template <typename T>
void launch_kernel(
    void *output,
    const void *start,
    const void *end,
    const void *weight,
    const LerpInfo &info,
    const LerpOpaqueData *opaque,
    void *stream) {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    auto out_ptr = reinterpret_cast<T *>(output);
    auto start_ptr = reinterpret_cast<const T *>(start);
    auto end_ptr = reinterpret_cast<const T *>(end);

    const T *weight_ptr = nullptr;
    float weight_scalar = 0.0f;

    if (info.is_scalar_weight()) {
        weight_scalar = info.weight_scalar();
    } else {
        weight_ptr = reinterpret_cast<const T *>(weight);
    }

    size_t numel = info.numel();
    int ndim = opaque->ndim;

    size_t block_size = 256;
    size_t grid_size = (numel + block_size - 1) / block_size;

    op::lerp::cuda::lerp_kernel<T>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            out_ptr,
            start_ptr,
            end_ptr,
            weight_ptr,
            weight_scalar,
            numel,
            ndim,
            opaque->d_shape,
            opaque->d_start_strides,
            opaque->d_end_strides,
            opaque->d_weight_strides);
}

// ==================================================================
// 4. Descriptor::create 实现
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t start_desc,
    infiniopTensorDescriptor_t end_desc,
    infiniopTensorDescriptor_t weight_desc,
    float weight_scalar) {

    // 引入了正确的头文件后，device::nvidia::Handle 现已可见
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto result = LerpInfo::create(out_desc, start_desc, end_desc, weight_desc, weight_scalar);
    CHECK_RESULT(result);
    auto info = result.take();

    auto opaque = new Opaque();
    opaque->ndim = static_cast<int>(out_desc->ndim());

    // 直接拷贝 vector
    const auto &shape_vec = out_desc->shape();
    std::vector<int64_t> host_shape(shape_vec.begin(), shape_vec.end());

    opaque->d_shape = upload_to_device(host_shape);

    std::vector<size_t> shape_dims(host_shape.begin(), host_shape.end());

    auto start_strides = compute_broadcast_strides(shape_dims, start_desc);
    opaque->d_start_strides = upload_to_device(start_strides);

    auto end_strides = compute_broadcast_strides(shape_dims, end_desc);
    opaque->d_end_strides = upload_to_device(end_strides);

    if (!info.is_scalar_weight() && weight_desc != nullptr) {
        auto weight_strides = compute_broadcast_strides(shape_dims, weight_desc);
        opaque->d_weight_strides = upload_to_device(weight_strides);
    }

    *desc_ptr = new Descriptor(
        opaque,
        info,
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 5. Descriptor::calculate 实现
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *start,
    const void *end,
    const void *weight,
    void *stream) const {

    auto dtype = _info.dtype();

    // _opaque 自动向上转型为 const LerpOpaqueData*
    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, start, end, weight, _info, _opaque, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(output, start, end, weight, _info, _opaque, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, start, end, weight, _info, _opaque, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, start, end, weight, _info, _opaque, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::lerp::nvidia
