#include "../../../devices/moore/moore_handle.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "binary_cross_entropy_with_logits_moore.h"
#include <musa_runtime.h>

namespace op::bce_with_logits::moore {

using device::moore::indexToOffset;

struct Descriptor::Opaque {};

Descriptor::~Descriptor() = default;

// 摩尔线程平台通常与 CUDA 保持一致的维度上限
constexpr int BCE_MAX_DIMS = 8;

struct BCETensorInfoDevice {
    size_t ndim;
    size_t shape[BCE_MAX_DIMS];
    ptrdiff_t strides[BCE_MAX_DIMS];
};

static inline BCETensorInfoDevice make_device_info(const BCETensorInfo &info) {
    BCETensorInfoDevice dev{};
    dev.ndim = info.ndim;
    for (size_t i = 0; i < info.ndim && i < static_cast<size_t>(BCE_MAX_DIMS); ++i) {
        dev.shape[i] = info.dims[i];
        dev.strides[i] = info.stride[i];
    }
    return dev;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t pos_weight_desc,
    infiniopReduction_t reduction) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = logits_desc->dtype();

    // Moore 实现支持 F16 / F32 / BF16
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = BCEWithLogitsInfo::create(out_desc, logits_desc, target_desc,
                                            weight_desc, pos_weight_desc, reduction);
    CHECK_RESULT(result);

    auto info = result.take();

    size_t workspace_size = 0;
    if (reduction != INFINIOP_REDUCTION_NONE && (dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16)) {
        workspace_size = sizeof(float);
    }

    *desc_ptr = new Descriptor(
        dtype, std::move(info), reduction, workspace_size,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// 针对 Moore 平台的类型提升逻辑
template <typename T>
__device__ __forceinline__ float to_float(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) { // MUSA 兼容 cuda_bfloat16 名称或使用内部 bf16
        return __bfloat162float(x);
    } else {
        return static_cast<float>(x);
    }
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(x);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16_rn(x); // Moore 平台推荐显式使用 _rn
    } else {
        return static_cast<T>(x);
    }
}

// --- MUSA Kernel ---
template <typename Tdata, typename Taccum>
__global__ void bce_logits_kernel(
    void *out_raw,
    const Tdata *logits,
    const Tdata *target,
    const Tdata *weight,
    const Tdata *pos_weight,
    BCETensorInfoDevice logits_info,
    BCETensorInfoDevice target_info,
    BCETensorInfoDevice weight_info,
    BCETensorInfoDevice out_info,
    size_t n,
    size_t pos_weight_len,
    infiniopReduction_t reduction) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    size_t logits_offset = indexToOffset(idx, logits_info.ndim,
                                         logits_info.shape, logits_info.strides);
    size_t target_offset = indexToOffset(idx, target_info.ndim,
                                         target_info.shape, target_info.strides);

    float x = to_float(logits[logits_offset]);
    float y = to_float(target[target_offset]);

    float pw = 1.0f;
    if (pos_weight && pos_weight_len > 0) {
        size_t c = idx % pos_weight_len;
        pw = to_float(pos_weight[c]);
    }

    float w = 1.0f;
    if (weight && weight_info.ndim > 0) {
        size_t weight_offset = indexToOffset(idx, weight_info.ndim,
                                             weight_info.shape, weight_info.strides);
        w = to_float(weight[weight_offset]);
    }

    // 数值稳定的 BCEWithLogits 计算（对齐 PyTorch 实现）：
    // max_val = max(-x, 0)
    // log_weight = 1 + (pos_weight - 1) * y
    // loss = (1 - y) * x + log_weight * (log(1 + exp(-|x|)) + max_val)
    float max_val = fmaxf(-x, 0.0f);
    float log_weight = 1.0f + (pw - 1.0f) * y;
    float loss = (1.0f - y) * x + log_weight * (logf(1.0f + expf(-fabsf(x))) + max_val);

    loss *= w;

    if (reduction == INFINIOP_REDUCTION_NONE) {
        size_t out_offset = indexToOffset(idx, out_info.ndim,
                                          out_info.shape, out_info.strides);
        auto *out_ptr = static_cast<Tdata *>(out_raw);
        out_ptr[out_offset] = from_float<Tdata>(loss);
    } else {
        auto *out_accum = static_cast<Taccum *>(out_raw);
        atomicAdd(out_accum, static_cast<Taccum>(loss));
    }
}

__global__ void bce_logits_mean_scale_kernel_f32(float *val, size_t count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *val /= static_cast<float>(count);
    }
}

template <typename Tdata>
__global__ void bce_logits_reduce_finalize_kernel(
    Tdata *out,
    float *workspace,
    size_t count,
    int is_mean) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float v = *workspace;
        if (is_mean) {
            v /= static_cast<float>(count);
        }
        *out = from_float<Tdata>(v);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *logits,
    const void *target,
    const void *weight,
    const void *pos_weight,
    void *stream) const {

    musaStream_t mustream = (musaStream_t)stream;
    size_t n = _info.num_elements;

    if (_reduction != INFINIOP_REDUCTION_NONE && (_dtype == INFINI_DTYPE_F16 || _dtype == INFINI_DTYPE_BF16)) {
        if (workspace_size < sizeof(float)) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
    }

    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);

    BCETensorInfoDevice logits_info = make_device_info(_info.logits);
    BCETensorInfoDevice target_info = make_device_info(_info.target);
    BCETensorInfoDevice out_info = make_device_info(_info.out);
    BCETensorInfoDevice weight_info = {};
    if (_info.weight.total_elements != 0) {
        weight_info = make_device_info(_info.weight);
    }

    size_t pos_weight_len = _info.pos_weight.total_elements;

    switch (_dtype) {
    case INFINI_DTYPE_F32: {
        if (_reduction != INFINIOP_REDUCTION_NONE) {
            musaMemsetAsync(out, 0, sizeof(float), mustream);
        }

        bce_logits_kernel<float, float><<<grid, block, 0, mustream>>>(
            out,
            static_cast<const float *>(logits),
            static_cast<const float *>(target),
            static_cast<const float *>(weight),
            static_cast<const float *>(pos_weight),
            logits_info,
            target_info,
            weight_info,
            out_info,
            n,
            pos_weight_len,
            _reduction);

        if (_reduction == INFINIOP_REDUCTION_MEAN) {
            bce_logits_mean_scale_kernel_f32<<<1, 1, 0, mustream>>>(
                static_cast<float *>(out), n);
        }
        break;
    }
    case INFINI_DTYPE_F16: {
        auto *logits_h = static_cast<const half *>(logits);
        auto *target_h = static_cast<const half *>(target);
        auto *weight_h = static_cast<const half *>(weight);
        auto *pos_weight_h = static_cast<const half *>(pos_weight);

        void *out_raw = nullptr;
        float *workspace_f = nullptr;

        if (_reduction == INFINIOP_REDUCTION_NONE) {
            out_raw = out;
        } else {
            workspace_f = static_cast<float *>(workspace);
            musaMemsetAsync(workspace_f, 0, sizeof(float), mustream);
            out_raw = workspace_f;
        }

        bce_logits_kernel<half, float><<<grid, block, 0, mustream>>>(
            out_raw,
            logits_h,
            target_h,
            weight_h,
            pos_weight_h,
            logits_info,
            target_info,
            weight_info,
            out_info,
            n,
            pos_weight_len,
            _reduction);

        if (_reduction != INFINIOP_REDUCTION_NONE) {
            int is_mean = (_reduction == INFINIOP_REDUCTION_MEAN) ? 1 : 0;
            bce_logits_reduce_finalize_kernel<half><<<1, 1, 0, mustream>>>(
                static_cast<half *>(out), workspace_f, n, is_mean);
        }
        break;
    }
    case INFINI_DTYPE_BF16: {
        auto *logits_b = static_cast<const cuda_bfloat16 *>(logits);
        auto *target_b = static_cast<const cuda_bfloat16 *>(target);
        auto *weight_b = static_cast<const cuda_bfloat16 *>(weight);
        auto *pos_weight_b = static_cast<const cuda_bfloat16 *>(pos_weight);

        void *out_raw = nullptr;
        float *workspace_f = nullptr;

        if (_reduction == INFINIOP_REDUCTION_NONE) {
            out_raw = out;
        } else {
            workspace_f = static_cast<float *>(workspace);
            musaMemsetAsync(workspace_f, 0, sizeof(float), mustream);
            out_raw = workspace_f;
        }

        bce_logits_kernel<cuda_bfloat16, float><<<grid, block, 0, mustream>>>(
            out_raw,
            logits_b,
            target_b,
            weight_b,
            pos_weight_b,
            logits_info,
            target_info,
            weight_info,
            out_info,
            n,
            pos_weight_len,
            _reduction);

        if (_reduction != INFINIOP_REDUCTION_NONE) {
            int is_mean = (_reduction == INFINIOP_REDUCTION_MEAN) ? 1 : 0;
            bce_logits_reduce_finalize_kernel<cuda_bfloat16><<<1, 1, 0, mustream>>>(
                static_cast<cuda_bfloat16 *>(out), workspace_f, n, is_mean);
        }
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::bce_with_logits::moore
