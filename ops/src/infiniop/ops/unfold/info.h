#ifndef __OPS_UNFOLD_INFO_H__
#define __OPS_UNFOLD_INFO_H__

// 按照你提供的路径引用
#include "../../../utils.h"
#include "../../tensor.h"

#include <numeric>
#include <vector>

namespace op::unfold {

struct UnfoldInfo {
public:
    int _dtype;

    // 空间参数
    std::vector<int> _kernel_sizes;
    std::vector<int> _strides;
    std::vector<int> _paddings;
    std::vector<int> _dilations;

    // 形状缓存
    size_t _N;     // Batch
    size_t _C_in;  // Input Channels
    size_t _C_out; // Output Channels
    size_t _L;     // Output Spatial Length

    // 缓存输入的空间维度 (H_in, W_in)
    std::vector<int64_t> _input_spatial_shape;
    // 缓存输出的空间维度 (H_out, W_out)
    std::vector<int64_t> _output_spatial_shape;

    // 默认构造必须是 public
    UnfoldInfo() = default;

    // 构造函数
    UnfoldInfo(int dtype,
               std::vector<int> kernel_sizes,
               std::vector<int> strides,
               std::vector<int> paddings,
               std::vector<int> dilations,
               size_t N, size_t C_in, size_t C_out, size_t L,
               std::vector<int64_t> in_spatial,
               std::vector<int64_t> out_spatial)
        : _dtype(dtype),
          _kernel_sizes(std::move(kernel_sizes)),
          _strides(std::move(strides)),
          _paddings(std::move(paddings)),
          _dilations(std::move(dilations)),
          _N(N), _C_in(C_in), _C_out(C_out), _L(L),
          _input_spatial_shape(std::move(in_spatial)),
          _output_spatial_shape(std::move(out_spatial)) {}

    // Getters
    int dtype_val() const { return _dtype; }

    // 对应 .cc 中的调用
    static utils::Result<UnfoldInfo> infer(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        const int *kernel_sizes,
        const int *strides,
        const int *paddings,
        const int *dilations) {

        // 1. 检查维度 (假设 ndim() 是方法)
        if (input_desc->ndim() < 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        int ndim = int(input_desc->ndim());
        int spatial_dims = ndim - 2;

        // 2. 读取参数
        size_t N = input_desc->shape()[0];
        size_t C_in = input_desc->shape()[1];

        std::vector<int> k_vec(kernel_sizes, kernel_sizes + spatial_dims);
        std::vector<int> s_vec(strides, strides + spatial_dims);
        std::vector<int> p_vec(paddings, paddings + spatial_dims);
        std::vector<int> d_vec(dilations, dilations + spatial_dims);

        // 3. 计算形状
        size_t kernel_prod = 1;
        for (int k : k_vec) {
            kernel_prod *= k;
        }
        size_t C_out = C_in * kernel_prod;

        size_t L = 1;
        std::vector<int64_t> in_spatial;
        std::vector<int64_t> out_spatial;

        for (int i = 0; i < spatial_dims; ++i) {
            int64_t in_dim = input_desc->shape()[i + 2];
            in_spatial.push_back(in_dim);

            int k = k_vec[i];
            int s = s_vec[i];
            int p = p_vec[i];
            int d = d_vec[i];

            int64_t numerator = in_dim + 2 * p - d * (k - 1) - 1;
            if (numerator < 0) {
                numerator = -s;
            }

            int64_t out_dim = (numerator / s) + 1;
            if (out_dim <= 0) {
                return INFINI_STATUS_BAD_PARAM;
            }

            out_spatial.push_back(out_dim);
            L *= out_dim;
        }

        // 4. 校验输出
        if (out_desc->ndim() == 3) {
            if (out_desc->shape()[0] != N || out_desc->shape()[1] != C_out || out_desc->shape()[2] != L) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 显式构造 Result
        return utils::Result<UnfoldInfo>(UnfoldInfo(
            input_desc->dtype(),
            std::move(k_vec), std::move(s_vec), std::move(p_vec), std::move(d_vec),
            N, C_in, C_out, L,
            std::move(in_spatial), std::move(out_spatial)));
    }
};

} // namespace op::unfold

#endif // __OPS_UNFOLD_INFO_H__
