#include "../../../elementwise/moore/elementwise_moore.h"
#include "addcmul_moore.h"
#include "addcmul_moore_kernel.h"
#include <musa_runtime.h>

namespace op::addcmul::moore {

Descriptor::~Descriptor() = default;

// 1. 填充 TensorMeta，逻辑与 NVIDIA 一致，用于 MUSA Kernel 中的 Strided 寻址
static inline infiniStatus_t fill_tensor_meta(
    infiniopTensorDescriptor_t desc,
    Descriptor::TensorMeta &meta) {

    auto ndim = desc->ndim();
    if (ndim > Descriptor::MAX_NDIM) {
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }

    meta.ndim = static_cast<int>(ndim);
    const auto &shape = desc->shape();
    const auto &strides = desc->strides();
    for (int i = 0; i < meta.ndim; ++i) {
        meta.shape[i] = shape[i];
        meta.strides[i] = strides[i];
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float value) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // 类型检查
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);

    // 形状检查 (A, T1, T2 需一致)
    const auto &out_shape = out_desc->shape();
    const auto &input_desc = input_desc_vec.at(0);
    const auto &t1_desc = input_desc_vec.at(1);
    const auto &t2_desc = input_desc_vec.at(2);
    CHECK_SAME_SHAPE(out_shape, input_desc->shape());
    CHECK_SAME_SHAPE(out_shape, t1_desc->shape());
    CHECK_SAME_SHAPE(out_shape, t2_desc->shape());

    // 2. 调用 Moore 平台的描述符创建宏
    CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    auto *desc = *desc_ptr;
    desc->_output_size = out_desc->numel();

    // 填充元数据
    CHECK_STATUS(fill_tensor_meta(out_desc, desc->_out_meta));
    CHECK_STATUS(fill_tensor_meta(input_desc, desc->_input_meta));
    CHECK_STATUS(fill_tensor_meta(t1_desc, desc->_t1_meta));
    CHECK_STATUS(fill_tensor_meta(t2_desc, desc->_t2_meta));

    desc->_value = value;

    return INFINI_STATUS_SUCCESS;
}

// 3. MUSA Kernel 实现：逻辑保持一致
template <typename T>
__global__ void addcmul_kernel(
    size_t output_size,
    Descriptor::TensorMeta out_meta,
    Descriptor::TensorMeta in_meta,
    Descriptor::TensorMeta t1_meta,
    Descriptor::TensorMeta t2_meta,
    T *out,
    const T *input,
    const T *t1,
    const T *t2,
    float value) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) {
        return;
    }

    ptrdiff_t out_offset = 0, in_offset = 0, t1_offset = 0, t2_offset = 0;
    size_t linear = idx;

    // 通用多维索引转偏移逻辑
    for (int dim = out_meta.ndim - 1; dim >= 0; --dim) {
        size_t dim_size = out_meta.shape[dim];
        size_t coord = linear % dim_size;
        linear /= dim_size;

        out_offset += static_cast<ptrdiff_t>(coord) * out_meta.strides[dim];
        in_offset += static_cast<ptrdiff_t>(coord) * in_meta.strides[dim];
        t1_offset += static_cast<ptrdiff_t>(coord) * t1_meta.strides[dim];
        t2_offset += static_cast<ptrdiff_t>(coord) * t2_meta.strides[dim];
    }

    // 调用 Moore 平台定义的 AddcmulOp
    out[out_offset] = op::addcmul::moore::AddcmulOp{}(input[in_offset], t1[t1_offset], t2[t2_offset], value);
}

// 4. 内核启动封装
template <typename T>
static inline infiniStatus_t launch_addcmul_kernel(
    const Descriptor *desc,
    void *output,
    const std::vector<const void *> &inputs,
    void *stream) {

    size_t output_size = desc->_output_size;
    if (output_size == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    auto *out_ptr = reinterpret_cast<T *>(output);
    auto *in_ptr = reinterpret_cast<const T *>(inputs.at(0));
    auto *t1_ptr = reinterpret_cast<const T *>(inputs.at(1));
    auto *t2_ptr = reinterpret_cast<const T *>(inputs.at(2));

    musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);

    constexpr uint32_t BLOCK_SIZE = 256;
    uint32_t grid = static_cast<uint32_t>((output_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    addcmul_kernel<T><<<grid, BLOCK_SIZE, 0, musa_stream>>>(
        output_size, desc->_out_meta, desc->_input_meta, desc->_t1_meta, desc->_t2_meta,
        out_ptr, in_ptr, t1_ptr, t2_ptr, desc->getValue());

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return launch_addcmul_kernel<half>(this, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        // 使用 Moore 平台对应的 bf16 类型
        return launch_addcmul_kernel<cuda_bfloat16>(this, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return launch_addcmul_kernel<float>(this, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return launch_addcmul_kernel<double>(this, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
} // namespace op::addcmul::moore
