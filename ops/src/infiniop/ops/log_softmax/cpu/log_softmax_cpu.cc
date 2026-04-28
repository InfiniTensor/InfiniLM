#include "log_softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::log_softmax::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int dim) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = LogSoftmaxInfo::create(output_desc, input_desc, dim);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void calculate_cpu_impl(
    const LogSoftmaxInfo &info,
    void *output,
    const void *input) {

    size_t outer_size = info.outer_size();
    size_t dim_size = info.dim_size();
    size_t inner_size = info.inner_size();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    size_t total_tasks = outer_size * inner_size;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t task_id = 0; task_id < (ptrdiff_t)total_tasks; ++task_id) {
        // 解算当前任务对应的外部索引和内部索引
        size_t o = task_id / inner_size;
        size_t i = task_id % inner_size;

        // 计算基地址偏移
        // Layout: [outer, dim, inner]
        // Base Offset = o * (dim_size * inner_size) + i
        size_t base_offset = o * dim_size * inner_size + i;
        size_t stride = inner_size;
        std::vector<float> buffer(dim_size);
        float max_val = -std::numeric_limits<float>::infinity();

        for (size_t d = 0; d < dim_size; ++d) {
            T val_t = in_ptr[base_offset + d * stride];
            float val = utils::cast<float>(val_t); // 处理 fp16/bf16
            buffer[d] = val;
            if (val > max_val) {
                max_val = val;
            }
        }

        //  计算指数和 (Sum)
        // Compute sum(exp(x - max))
        float sum_exp = 0.0f;
        for (size_t d = 0; d < dim_size; ++d) {
            sum_exp += std::exp(buffer[d] - max_val);
        }

        // 计算 LogSumExp
        // log(sum(e^(x-M))) + M
        float log_sum_exp = std::log(sum_exp) + max_val;

        //  计算最终结果并写入
        // output = x - LogSumExp
        for (size_t d = 0; d < dim_size; ++d) {
            float res = buffer[d] - log_sum_exp;
            out_ptr[base_offset + d * stride] = utils::cast<T>(res);
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, input);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::log_softmax::cpu
