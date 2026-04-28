#ifndef __LDEXP_INFO_H__
#define __LDEXP_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace op::ldexp {

class LdexpInfo {
    LdexpInfo() = default;

public:
    int _dtype;
    int _exp_dtype; // [新增] 记录指数的数据类型
    size_t _count;

    // 维度、形状和广播步长
    int _ndim;
    std::vector<int> _shape;
    std::vector<int> _x_strides;
    std::vector<int> _exp_strides;

    int dtype() const { return _dtype; }
    int exp_dtype() const { return _exp_dtype; } // [新增] Getter
    size_t count() const { return _count; }

    // Accessors
    int ndim() const { return _ndim; }
    const std::vector<int> &shape() const { return _shape; }
    const std::vector<int> &x_strides() const { return _x_strides; }
    const std::vector<int> &exp_strides() const { return _exp_strides; }

    // [修改] 更新构造函数，增加 exp_dtype
    LdexpInfo(int dtype, int exp_dtype, size_t count, int ndim,
              std::vector<int> shape,
              std::vector<int> x_strides,
              std::vector<int> exp_strides)
        : _dtype(dtype), _exp_dtype(exp_dtype), _count(count), _ndim(ndim),
          _shape(std::move(shape)),
          _x_strides(std::move(x_strides)),
          _exp_strides(std::move(exp_strides)) {}

    static utils::Result<LdexpInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t exp_desc) {

        if (y_desc->dtype() != x_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        int dtype = x_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32 && dtype != INFINI_DTYPE_F64 && dtype != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        int exp_dtype = exp_desc->dtype();
        if (exp_dtype != dtype && exp_dtype != INFINI_DTYPE_I32 && exp_dtype != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        int ndim_y = int(y_desc->ndim());
        int ndim_x = int(x_desc->ndim());
        int ndim_exp = int(exp_desc->ndim());

        int ndim_out = std::max(ndim_x, ndim_exp);

        if (ndim_y != ndim_out) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 准备存储形状和步长
        std::vector<int> shape(ndim_out);
        std::vector<int> x_strides(ndim_out);
        std::vector<int> exp_strides(ndim_out);

        size_t total_count = 1;

        // ---------------------------------------------------------
        // 1. 确定输出形状 (Shape Inference)
        // ---------------------------------------------------------
        for (int i = 0; i < ndim_out; ++i) {
            int x_dim_idx = i - (ndim_out - ndim_x);
            int exp_dim_idx = i - (ndim_out - ndim_exp);

            size_t dim_x = (x_dim_idx >= 0) ? x_desc->shape()[x_dim_idx] : 1;
            size_t dim_exp = (exp_dim_idx >= 0) ? exp_desc->shape()[exp_dim_idx] : 1;

            if (dim_x != dim_exp && dim_x != 1 && dim_exp != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            size_t expected_dim_y = std::max(dim_x, dim_exp);

            if (y_desc->shape()[i] != expected_dim_y) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            shape[i] = static_cast<int>(expected_dim_y);
            total_count *= expected_dim_y;
        }

        // ---------------------------------------------------------
        // 2. 计算广播步长 (Compute Broadcasting Strides)
        // ---------------------------------------------------------
        auto compute_strides = [&](infiniopTensorDescriptor_t input_desc, std::vector<int> &out_strides) {
            int input_ndim = int(input_desc->ndim());
            int offset = ndim_out - input_ndim;

            std::vector<int> dense_strides(input_ndim);
            int current_stride = 1;
            for (int i = input_ndim - 1; i >= 0; --i) {
                dense_strides[i] = current_stride;
                current_stride *= int(input_desc->shape()[i]);
            }

            for (int i = 0; i < ndim_out; ++i) {
                if (i < offset) {
                    out_strides[i] = 0;
                } else {
                    int input_dim_idx = i - offset;
                    int input_dim_size = int(input_desc->shape()[input_dim_idx]);

                    if (input_dim_size == 1 && shape[i] > 1) {
                        out_strides[i] = 0;
                    } else {
                        out_strides[i] = dense_strides[input_dim_idx];
                    }
                }
            }
        };

        compute_strides(x_desc, x_strides);
        compute_strides(exp_desc, exp_strides);

        return utils::Result<LdexpInfo>(LdexpInfo{
            dtype,
            exp_dtype, // [修改] 传递 exp_dtype
            total_count,
            ndim_out,
            shape,
            x_strides,
            exp_strides});
    }
};

} // namespace op::ldexp

#endif
