#ifndef __ALL_INFO_H__
#define __ALL_INFO_H__
#include "../../../utils.h"
#include "../../tensor.h"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace op::all {
class AllInfo {
    AllInfo() = default;

public:
    infiniDtype_t dtype;
    std::vector<size_t> permuted_input_shape; // need to permute
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> permuted_input_strides; // need to permute
    std::vector<ptrdiff_t> output_strides;
    size_t reduce_dim_size; // reduce dim size
    size_t reduce_num;      // number of elements to reduce for each output element
    size_t input_size;      // total number of input elements
    size_t output_size;     // total number of output elements
    size_t ndim;            // number of dimensions
    static utils::Result<AllInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        size_t *dim,
        size_t dim_size,
        bool keepdim) {
        auto input_shape = input_desc->shape();
        auto input_strides = input_desc->strides();
        size_t input_ndim = input_desc->ndim();
        size_t reduce_num = 1;
        for (size_t i = 0; i < dim_size; i++) {
            reduce_num *= input_shape[dim[i]];
        }
        std::vector<size_t> permute_order;
        for (size_t i = 0; i < input_ndim; i++) {
            if (std::find(dim, dim + dim_size, i) == dim + dim_size) {
                permute_order.push_back(i);
            }
        }
        for (size_t i = 0; i < dim_size; i++) {
            permute_order.push_back(dim[i]);
        }
        std::vector<size_t> permuted_input_shape;
        std::vector<ptrdiff_t> permuted_input_strides;
        for (size_t i = 0; i < permute_order.size(); i++) {
            permuted_input_shape.push_back(input_shape[permute_order[i]]);
            permuted_input_strides.push_back(input_strides[permute_order[i]]);
        }
        return utils::Result<AllInfo>(AllInfo{input_desc->dtype(),
                                              permuted_input_shape,
                                              output_desc->shape(),
                                              permuted_input_strides,
                                              output_desc->strides(),
                                              dim_size,
                                              reduce_num,
                                              input_desc->numel(),
                                              output_desc->numel(),
                                              input_ndim});
    }
};
} // namespace op::all

#endif
