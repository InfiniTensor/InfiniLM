#include "kthvalue_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <type_traits> // 引入 type_traits 以支持 constexpr 判断
#include <utility>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::kthvalue::cpu {

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
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t input_desc,
    int k,
    int dim,
    int keepdim) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = KthvalueInfo::create(values_desc, indices_desc, input_desc, k, dim, keepdim);
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
    const KthvalueInfo &info,
    void *values,
    void *indices,
    const void *input) {

    size_t outer_size = info.outer_size();
    size_t dim_size = info.dim_size();
    size_t inner_size = info.inner_size();
    int k = info.k(); // k is 1-based

    auto val_ptr = reinterpret_cast<T *>(values);
    auto idx_ptr = reinterpret_cast<int64_t *>(indices);
    auto in_ptr = reinterpret_cast<const T *>(input);

    size_t total_tasks = outer_size * inner_size;

    // k 在输入中是 1-based，转为 0-based 用于 vector索引
    int k_idx = k - 1;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t task_id = 0; task_id < (ptrdiff_t)total_tasks; ++task_id) {
        // 解算当前任务对应的外部索引和内部索引
        size_t o = task_id / inner_size;
        size_t i = task_id % inner_size;

        // 计算输入数据的基地址偏移
        // Input layout logic: [outer, dim, inner]
        // Offset = o * (dim_size * inner_size) + [0...dim_size-1] * inner_size + i
        size_t input_base_offset = o * dim_size * inner_size + i;
        size_t stride = inner_size;

        // 使用临时容器存储 (数值, 原始索引)
        // 注意：这里在循环内分配内存，由于 dim_size 通常不大，对 CPU 来说尚可接受
        std::vector<std::pair<T, int64_t>> row_data;
        row_data.reserve(dim_size);

        for (size_t d = 0; d < dim_size; ++d) {
            T val = in_ptr[input_base_offset + d * stride];
            row_data.push_back({val, static_cast<int64_t>(d)});
        }

        // 使用 nth_element 找到第 k 小的元素 (O(N) 复杂度)
        // 修复: 使用 utils::cast<float> 确保自定义类型(fp16/bf16)可以比较
        std::nth_element(
            row_data.begin(),
            row_data.begin() + k_idx,
            row_data.end(),
            [](const std::pair<T, int64_t> &a, const std::pair<T, int64_t> &b) {
                // 如果是标准算术类型，直接比较；如果是自定义类型，转换为 float 比较
                if constexpr (std::is_arithmetic_v<T>) {
                    return a.first < b.first;
                } else {
                    return utils::cast<float>(a.first) < utils::cast<float>(b.first);
                }
            });

        // 获取结果
        auto result_pair = row_data[k_idx];

        // 写入输出
        val_ptr[task_id] = result_pair.first;
        idx_ptr[task_id] = result_pair.second;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *values,
    void *indices,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, values, indices, input);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, values, indices, input);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, values, indices, input);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, values, indices, input);
        break;
    case INFINI_DTYPE_I32:
        cpu::calculate_cpu_impl<int32_t>(_info, values, indices, input);
        break;
    case INFINI_DTYPE_I64:
        cpu::calculate_cpu_impl<int64_t>(_info, values, indices, input);
        break;
    case INFINI_DTYPE_U32:
        cpu::calculate_cpu_impl<uint32_t>(_info, values, indices, input);
        break;
    case INFINI_DTYPE_U64:
        cpu::calculate_cpu_impl<uint64_t>(_info, values, indices, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::kthvalue::cpu
