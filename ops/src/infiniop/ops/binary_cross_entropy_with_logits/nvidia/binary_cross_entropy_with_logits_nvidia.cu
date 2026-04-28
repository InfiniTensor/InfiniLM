#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "binary_cross_entropy_with_logits_nvidia.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>

namespace op::bce_with_logits::nvidia {

using device::nvidia::indexToOffset;

struct Descriptor::Opaque {};

Descriptor::~Descriptor() = default;

// 在 GPU 侧使用的简化张量信息（固定上限维度，支持 stride）
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

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = logits_desc->dtype();

    // NVIDIA 实现支持 F16 / F32 / BF16
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = BCEWithLogitsInfo::create(out_desc, logits_desc, target_desc,
                                            weight_desc, pos_weight_desc, reduction);
    CHECK_RESULT(result);

    auto info = result.take();

    // F16/BF16 在做归约时需要一个 float 标量 workspace 来累加
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

// 将任意标量类型提升为 float
template <typename T>
__device__ __forceinline__ float to_float(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __bfloat162float(x);
    } else {
        return static_cast<float>(x);
    }
}

// 从 float 转回目标标量类型
template <typename T>
__device__ __forceinline__ T from_float(float x) {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(x);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16(x);
    } else {
        return static_cast<T>(x);
    }
}

// --- CUDA Kernel: 支持 stride 的数值稳定 BCE 计算 ---
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

    // 计算逻辑索引在各张量中的偏移（支持任意 stride）
    size_t logits_offset = indexToOffset(idx, logits_info.ndim,
                                         logits_info.shape, logits_info.strides);
    size_t target_offset = indexToOffset(idx, target_info.ndim,
                                         target_info.shape, target_info.strides);

    float x = to_float(logits[logits_offset]);
    float y = to_float(target[target_offset]);

    float pw = 1.0f;
    if (pos_weight && pos_weight_len > 0) {
        // 按最后一维广播：假设 pos_weight 是连续的一维张量
        size_t c = idx % pos_weight_len;
        pw = to_float(pos_weight[c]);
    }

    float w = 1.0f;
    if (weight && weight_info.ndim > 0) {
        size_t weight_offset = indexToOffset(idx, weight_info.ndim,
                                             weight_info.shape, weight_info.strides);
        w = to_float(weight[weight_offset]);
    }

    // 数值稳定公式：max(x, 0) - x * y * pw + (1 + (pw - 1) * y) * log(1 + exp(-abs(x)))
    // max_val = max(-x, 0)
    // log_weight = 1 + (pos_weight - 1) * y
    // loss = (1 - y) * x + log_weight * (log1p(exp(-|x|)) + max_val)
    float max_val = fmaxf(-x, 0.0f);
    float log_weight = 1.0f + (pw - 1.0f) * y;
    float loss = (1.0f - y) * x + log_weight * (log1pf(expf(-fabsf(x))) + max_val);
    loss *= w;
    if (reduction == INFINIOP_REDUCTION_NONE) {
        // 写回逐元素 loss（支持 stride 的 out）
        size_t out_offset = indexToOffset(idx, out_info.ndim,
                                          out_info.shape, out_info.strides);
        auto *out_ptr = static_cast<Tdata *>(out_raw);
        out_ptr[out_offset] = from_float<Tdata>(loss);
    } else {
        // 对于 mean 或 sum，使用 float 累加到标量位置
        auto *out_accum = static_cast<Taccum *>(out_raw);
        atomicAdd(out_accum, static_cast<Taccum>(loss));
    }
}

// F32 mean 归约：对输出标量做除法
__global__ void bce_logits_mean_scale_kernel_f32(float *val, size_t count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *val /= static_cast<float>(count);
    }
}

// F16/BF16 归约：从 float workspace 写回目标 dtype
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

    cudaStream_t custream = (cudaStream_t)stream;
    size_t n = _info.num_elements;

    // F16/BF16 + 归约需要 float workspace
    if (_reduction != INFINIOP_REDUCTION_NONE && (_dtype == INFINI_DTYPE_F16 || _dtype == INFINI_DTYPE_BF16)) {
        if (workspace_size < sizeof(float)) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
    }

    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);

    // 构造 GPU 侧的张量信息（含 stride）
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
        // 如果是规约操作，计算前需将输出位置清零
        if (_reduction != INFINIOP_REDUCTION_NONE) {
            cudaMemsetAsync(out, 0, sizeof(float), custream);
        }

        bce_logits_kernel<float, float><<<grid, block, 0, custream>>>(
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
            bce_logits_mean_scale_kernel_f32<<<1, 1, 0, custream>>>(
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
            cudaMemsetAsync(workspace_f, 0, sizeof(float), custream);
            out_raw = workspace_f;
        }

        bce_logits_kernel<half, float><<<grid, block, 0, custream>>>(
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
            bce_logits_reduce_finalize_kernel<half><<<1, 1, 0, custream>>>(
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
            cudaMemsetAsync(workspace_f, 0, sizeof(float), custream);
            out_raw = workspace_f;
        }

        bce_logits_kernel<cuda_bfloat16, float><<<grid, block, 0, custream>>>(
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
            bce_logits_reduce_finalize_kernel<cuda_bfloat16><<<1, 1, 0, custream>>>(
                static_cast<cuda_bfloat16 *>(out), workspace_f, n, is_mean);
        }

        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto err = cudaGetLastError();
    return (err == cudaSuccess) ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::bce_with_logits::nvidia
