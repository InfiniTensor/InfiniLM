#ifndef __SOFTPLUS_INFO_H__
#define __SOFTPLUS_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <algorithm> // for std::equal
#include <vector>

namespace op::softplus {

class SoftplusInfo {
    SoftplusInfo() = default;

public:
    int _dtype;           // 数据类型
    float _beta;          // 缩放参数
    float _threshold;     // 阈值参数
    size_t _num_elements; // 元素总数

    // [新增] 内存布局信息
    bool _is_contiguous;           // 是否内存连续
    std::vector<int64_t> _shape;   // 形状
    std::vector<int64_t> _strides; // 步长

    int dtype() const { return _dtype; }
    float beta() const { return _beta; }
    float threshold() const { return _threshold; }
    size_t num_elements() const { return _num_elements; }

    // [新增] Getters
    bool is_contiguous() const { return _is_contiguous; }
    const std::vector<int64_t> &shape() const { return _shape; }
    const std::vector<int64_t> &strides() const { return _strides; }
    int ndim() const { return int(_shape.size()); }

    // 构造函数
    SoftplusInfo(int dtype, float beta, float threshold, size_t num_elements,
                 bool is_contiguous, std::vector<int64_t> shape, std::vector<int64_t> strides)
        : _dtype(dtype), _beta(beta), _threshold(threshold), _num_elements(num_elements),
          _is_contiguous(is_contiguous), _shape(std::move(shape)), _strides(std::move(strides)) {}

    static utils::Result<SoftplusInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        float beta,
        float threshold) {

        if (out_desc->dtype() != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->ndim() != input_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 1. 检查形状一致性并计算元素总数
        // 注意：这里假设 shape() 返回的是支持下标访问的容器 (如 vector)
        auto out_shape = out_desc->shape();
        auto in_shape = input_desc->shape();
        size_t num_elements = 1;

        for (size_t i = 0; i < input_desc->ndim(); ++i) {
            if (out_shape[i] != in_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            num_elements *= in_shape[i];
        }

        // 2. [关键修复] 提取形状和步长
        // input_desc->shape() 返回 vector，必须用迭代器初始化，不能用指针加法
        auto in_strides = input_desc->strides();

        // 使用迭代器区间构造，同时自动处理从 uint64 到 int64 的隐式转换
        std::vector<int64_t> shape(in_shape.begin(), in_shape.end());
        std::vector<int64_t> strides(in_strides.begin(), in_strides.end());

        int ndim = int(shape.size());

        // 3. 检查连续性
        // 从后往前检查：stride[i] == stride[i+1] * shape[i+1]
        bool is_contiguous = true;
        int64_t expected_stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            if (shape[i] > 1) {
                if (strides[i] != expected_stride) {
                    is_contiguous = false;
                    break;
                }
                expected_stride *= shape[i];
            }
        }

        return utils::Result<SoftplusInfo>(SoftplusInfo{
            input_desc->dtype(), // _dtype
            beta,                // _beta
            threshold,           // _threshold
            num_elements,        // _num_elements
            is_contiguous,       // _is_contiguous
            shape,               // _shape
            strides              // _strides
        });
    }
};

} // namespace op::softplus

#endif // __SOFTPLUS_INFO_H__
