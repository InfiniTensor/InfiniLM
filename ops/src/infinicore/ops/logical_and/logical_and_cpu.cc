#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/logical_and.hpp"
#include "infinicore/tensor.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace infinicore::op::logical_and_impl::cpu {

void calculate(Tensor output, Tensor input1, Tensor input2) {
    auto ndim = output->ndim();
    auto numel = output->numel();
    auto shapes = output->shape();

    auto strides1 = input1->strides();
    auto strides2 = input2->strides();
    auto out_strides = output->strides();

    auto dtype = input1->dtype();
    auto dtype_size = input1->element_size();

    auto out_dtype = output->dtype();
    auto out_dtype_size = output->element_size();

    // 假设 Tensor::data() 返回的是支持字节加法的指针 (char* 或 uint8_t*)
    // 如果是 void*，建议显式强转为 uint8_t* 或 char*
    auto input1_base = reinterpret_cast<uint8_t *>(input1->data());
    auto input2_base = reinterpret_cast<uint8_t *>(input2->data());
    auto output_base = reinterpret_cast<uint8_t *>(output->data());

    std::vector<size_t> indices(ndim, 0);

    for (size_t idx = 0; idx < numel; ++idx) {
        size_t offset1 = 0;
        size_t offset2 = 0;
        size_t out_offset = 0;

        for (size_t dim = 0; dim < ndim; ++dim) {
            offset1 += indices[dim] * strides1[dim];
            offset2 += indices[dim] * strides2[dim];
            out_offset += indices[dim] * out_strides[dim];
        }

        bool result = false;

        // ==========================================
        //  INPUT TYPE DISPATCH (输入类型分发)
        // ==========================================

        // 1. 浮点型
        if (dtype == DataType::F32) {
            auto *p1 = reinterpret_cast<float *>(input1_base + offset1 * dtype_size);
            auto *p2 = reinterpret_cast<float *>(input2_base + offset2 * dtype_size);
            result = (*p1 != 0.0f) && (*p2 != 0.0f);
        } else if (dtype == DataType::F64) {
            auto *p1 = reinterpret_cast<double *>(input1_base + offset1 * dtype_size);
            auto *p2 = reinterpret_cast<double *>(input2_base + offset2 * dtype_size);
            result = (*p1 != 0.0) && (*p2 != 0.0);
        } else if (dtype == DataType::F16) {
            auto *p1 = reinterpret_cast<fp16_t *>(input1_base + offset1 * dtype_size);
            auto *p2 = reinterpret_cast<fp16_t *>(input2_base + offset2 * dtype_size);
            float v1 = utils::cast<float>(*p1);
            float v2 = utils::cast<float>(*p2);
            result = (v1 != 0.0f) && (v2 != 0.0f);

            // 2. 布尔与8位整型
        } else if (dtype == DataType::BOOL || dtype == DataType::U8) {
            auto *p1 = reinterpret_cast<uint8_t *>(input1_base + offset1 * dtype_size);
            auto *p2 = reinterpret_cast<uint8_t *>(input2_base + offset2 * dtype_size);
            result = (*p1 != 0) && (*p2 != 0);

            // 3. 【新增】32位整型 (修复 int32 测试失败的关键！)
        } else if (dtype == DataType::I32 || dtype == DataType::U32) {
            // 无论是 I32 还是 U32，做非零判断逻辑是一样的，直接强转成 int32_t 读取即可
            auto *p1 = reinterpret_cast<int32_t *>(input1_base + offset1 * dtype_size);
            auto *p2 = reinterpret_cast<int32_t *>(input2_base + offset2 * dtype_size);
            result = (*p1 != 0) && (*p2 != 0);

            // 4. 【新增】64位整型 (增强健壮性)
        } else if (dtype == DataType::I64 || dtype == DataType::U64) {
            auto *p1 = reinterpret_cast<int64_t *>(input1_base + offset1 * dtype_size);
            auto *p2 = reinterpret_cast<int64_t *>(input2_base + offset2 * dtype_size);
            result = (*p1 != 0) && (*p2 != 0);

            // 5. 【新增】16位整型 (增强健壮性)
        } else if (dtype == DataType::I16 || dtype == DataType::U16) {
            auto *p1 = reinterpret_cast<int16_t *>(input1_base + offset1 * dtype_size);
            auto *p2 = reinterpret_cast<int16_t *>(input2_base + offset2 * dtype_size);
            result = (*p1 != 0) && (*p2 != 0);

        } else {
            // 如果遇到 I8，可以合并到 U8 处理；如果没有，这里抛出异常是正确的
            throw std::runtime_error("Unsupported data type for logical_and operation.");
        }

        // ==========================================
        //  OUTPUT TYPE DISPATCH (输出类型分发)
        // ==========================================
        if (out_dtype == DataType::BOOL || out_dtype == DataType::U8) {
            auto *output_ptr = reinterpret_cast<uint8_t *>(output_base + out_offset * out_dtype_size);
            *output_ptr = result ? 1 : 0;
        } else if (out_dtype == DataType::F32) {
            *reinterpret_cast<float *>(output_base + out_offset * out_dtype_size) = result ? 1.0f : 0.0f;
        } else if (out_dtype == DataType::I32) { // 预防性增加对 int32 输出的支持
            *reinterpret_cast<int32_t *>(output_base + out_offset * out_dtype_size) = result ? 1 : 0;
        } else {
            // 也可以选择在这里 throw，或者默认不做处理
        }

        // --- 维度索引递增逻辑 ---
        for (ptrdiff_t dim = ndim - 1; dim >= 0; --dim) {
            indices[dim]++;
            if (indices[dim] < shapes[dim]) {
                break;
            } else {
                indices[dim] = 0;
            }
        }
    }
}

static bool registered = []() {
    LogicalAnd::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::logical_and_impl::cpu
