#include "addr_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::addr::cpu {
Descriptor::~Descriptor() = default;

// Template function to handle different data types
template <typename Tdata>
infiniStatus_t addr_impl(Tdata *out,
                         const Tdata *input,
                         const Tdata *vec1,
                         const Tdata *vec2,
                         const AddrInfo &info,
                         void *workspace,
                         size_t workspace_size) {
    size_t n = info.vec1_size;
    size_t m = info.vec2_size;
    float beta = info.beta;
    float alpha = info.alpha;

    size_t total = n * m;

#pragma omp parallel for
    for (ptrdiff_t idx = 0; idx < (ptrdiff_t)total; ++idx) {

        // 🔹 Decode (i, j)
        size_t i = idx / m;
        size_t j = idx % m;

        size_t v1_idx = i * info.vec1_stride;
        size_t v2_idx = j * info.vec2_stride;
        size_t in_idx = i * info.input_stride0 + j * info.input_stride1;
        size_t out_idx = i * info.output_stride0 + j * info.output_stride1;

        if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {

            float a = utils::cast<float>(vec1[v1_idx]);
            float b = utils::cast<float>(vec2[v2_idx]);
            float c = utils::cast<float>(input[in_idx]);

            out[out_idx] = utils::cast<Tdata>(alpha * a * b + beta * c);

        } else {

            float a = (float)vec1[v1_idx];
            float b = (float)vec2[v2_idx];
            float c = (float)input[in_idx];

            out[out_idx] = utils::cast<Tdata>(alpha * a * b + beta * c);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t vec1_desc,
    infiniopTensorDescriptor_t vec2_desc,
    float beta,
    float alpha) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = AddrInfo::create(input_desc, out_desc, vec1_desc, vec2_desc, beta, alpha);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(info.take(), 0, nullptr,
                               INFINI_DEVICE_CPU, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *input,
    const void *vec1,
    const void *vec2,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_F32:
        return addr_impl(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(input),
                         reinterpret_cast<const float *>(vec1),
                         reinterpret_cast<const float *>(vec2),
                         _info, workspace, workspace_size);
        break;
    case INFINI_DTYPE_F16:
        return addr_impl(reinterpret_cast<fp16_t *>(out),
                         reinterpret_cast<const fp16_t *>(input),
                         reinterpret_cast<const fp16_t *>(vec1),
                         reinterpret_cast<const fp16_t *>(vec2),
                         _info, workspace, workspace_size);
    case INFINI_DTYPE_BF16:
        return addr_impl(reinterpret_cast<bf16_t *>(out),
                         reinterpret_cast<const bf16_t *>(input),
                         reinterpret_cast<const bf16_t *>(vec1),
                         reinterpret_cast<const bf16_t *>(vec2),
                         _info, workspace, workspace_size);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::addr::cpu
