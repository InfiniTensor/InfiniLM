#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h" // Handle 定义

#include "../cuda/kernel.cuh"        // Descriptor 基类定义 (关键！)
#include "addbmm_nvidia.cuh"         // Tiled Kernel Launcher 定义
#include "infinicore/ops/addbmm.hpp" // Descriptor 声明
#include <vector>

// ==================================================================
// 匿名命名空间：辅助函数 Wrapper
// ==================================================================
namespace {

// 引用 Info 类
using AddbmmInfo = ::op::addbmm::AddbmmInfo;

// 泛型 Wrapper：负责从 Info 提取参数并调用底层 Launcher
template <typename T>
void launch_kernel_wrapper(
    void *output,
    const void *input,
    const void *batch1,
    const void *batch2,
    const AddbmmInfo &info, // 接收 Info 对象
    void *stream) {

    // 1. 提取维度
    size_t b = info.b();
    size_t n = info.n();
    size_t m = info.m();
    size_t p = info.p();
    float alpha = info.alpha();
    float beta = info.beta();

    // 2. 提取 Strides
    const auto &os = info.out_strides();
    const auto &is = info.in_strides();
    const auto &b1s = info.b1_strides();
    const auto &b2s = info.b2_strides();

    // 3. 调用 .cuh 中的优化版 Launcher
    // 【关键修复】不再使用 addbmm_kernel<<<...>>>
    // 而是调用 op::addbmm::nvidia::launch_kernel
    ::op::addbmm::nvidia::launch_addbmm_naive<T>(
        output, input, batch1, batch2,
        b, n, m, p,
        alpha, beta,
        // 显式转换为 ptrdiff_t，匹配 .cuh 签名
        static_cast<ptrdiff_t>(os[0]), static_cast<ptrdiff_t>(os[1]),
        static_cast<ptrdiff_t>(is[0]), static_cast<ptrdiff_t>(is[1]),
        static_cast<ptrdiff_t>(b1s[0]), static_cast<ptrdiff_t>(b1s[1]), static_cast<ptrdiff_t>(b1s[2]),
        static_cast<ptrdiff_t>(b2s[0]), static_cast<ptrdiff_t>(b2s[1]), static_cast<ptrdiff_t>(b2s[2]),
        stream);
}

} // anonymous namespace

// ==================================================================
// Descriptor 成员函数实现
// ==================================================================
namespace op::addbmm::nvidia {

// Opaque 结构体定义
struct Descriptor::Opaque {};

// 析构函数
Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

// Create 函数实现
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec, // 接收 Vector
    float alpha,
    float beta) {

    // 1. 参数校验
    if (input_desc_vec.size() != 3) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // 2. 调用 Info::create 解析参数
    auto info_result = ::op::addbmm::AddbmmInfo::create(
        out_desc,
        input_desc_vec[0], // input
        input_desc_vec[1], // batch1
        input_desc_vec[2], // batch2
        alpha,
        beta);

    if (!info_result) {
        return info_result.status();
    }

    // 3. 创建 Descriptor 实例
    *desc_ptr = new Descriptor(
        new Opaque(),
        info_result.take(),
        0, // Tiled Kernel 不需要 workspace
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// Calculate 函数实现
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // 1. 参数校验
    if (inputs.size() != 3) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const void *input_ptr = inputs[0];
    const void *batch1_ptr = inputs[1];
    const void *batch2_ptr = inputs[2];

    // 2. 提取参数
    auto dtype = _info.dtype();

    // 3. 分发 Kernel
    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel_wrapper<half>(
            output, input_ptr, batch1_ptr, batch2_ptr, _info, stream);
        break;

    case INFINI_DTYPE_BF16:
        launch_kernel_wrapper<cuda_bfloat16>(
            output, input_ptr, batch1_ptr, batch2_ptr, _info, stream);
        break;

    case INFINI_DTYPE_F32:
        launch_kernel_wrapper<float>(
            output, input_ptr, batch1_ptr, batch2_ptr, _info, stream);
        break;

    case INFINI_DTYPE_F64:
        // 假设 double 也使用 Tiled Kernel (如果 .cuh 支持)
        launch_kernel_wrapper<double>(
            output, input_ptr, batch1_ptr, batch2_ptr, _info, stream);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::addbmm::nvidia
