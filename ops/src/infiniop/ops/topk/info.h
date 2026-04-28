#ifndef __TOPK_INFO_H__
#define __TOPK_INFO_H__
#include "../../../utils.h"
#include "../../tensor.h"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace op::topk {
class TopKInfo {
    TopKInfo() = default;

public:
    infiniDtype_t dtype;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    size_t k;
    size_t dim;
    bool largest;
    bool sorted;
    size_t ndim;
    size_t dim_elements; // processed dim elements
    size_t n_iteration;  // total number of topk iteration
    static utils::Result<TopKInfo> create(
        infiniopTensorDescriptor_t values_output_desc,
        infiniopTensorDescriptor_t indices_output_desc,
        infiniopTensorDescriptor_t input_desc,
        size_t k,
        size_t dim,
        bool largest,
        bool sorted) {
        auto input_shape = input_desc->shape();
        auto input_strides = input_desc->strides();
        size_t input_ndim = input_desc->ndim();
        size_t dim_elements = input_shape[dim];
        size_t n_iteration = 1;
        for (size_t i = 0; i < input_ndim; i++) {
            if (i != dim) {
                n_iteration *= input_shape[i];
            }
        }
        return utils::Result<TopKInfo>(TopKInfo{input_desc->dtype(),
                                                input_desc->shape(),
                                                values_output_desc->shape(),
                                                input_desc->strides(),
                                                values_output_desc->strides(),
                                                k,
                                                dim,
                                                largest,
                                                sorted,
                                                input_ndim,
                                                dim_elements,
                                                n_iteration});
    }
};
} // namespace op::topk

#endif
