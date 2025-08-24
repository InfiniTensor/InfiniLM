#include "../tensor.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

std::shared_ptr<Tensor> Tensor::sliceImpl(const std::vector<SliceParams> &slices) const {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();

    auto new_shape = std::vector<size_t>(this->shape());
    ptrdiff_t offset = 0;

    for (const auto &slice : slices) {
        ASSERT(slice.len > 0);
        // Print error information for dimension and slice
        if (slice.start >= this->shape()[slice.dim]) {
            std::cerr << "Error: slice start (" << slice.start << ") >= dimension size (" 
                  << this->shape()[slice.dim] << ") for dimension " << slice.dim << std::endl;
        }
        if (this->shape()[slice.dim] < slice.start + slice.len) {
            std::cerr << "Error: slice end (" << slice.start + slice.len << ") > dimension size (" 
                  << this->shape()[slice.dim] << ") for dimension " << slice.dim << std::endl;
        }

        ASSERT(this->shape()[slice.dim] >= slice.start + slice.len);
        new_shape[slice.dim] = slice.len;
        offset += slice.start * this->strides()[slice.dim];
    }

    tensor->_desc = TensorDesc::create(this->dtype(), new_shape, this->strides());
    tensor->_offset = offset * dsize(this->dtype()) + this->_offset;
    tensor->_storage = this->_storage;
    return tensor;
}

std::shared_ptr<Tensor> Tensor::slice(size_t dim, size_t start, size_t len) {
    return this->sliceImpl({{dim, start, len}});
}

std::shared_ptr<Tensor const> Tensor::slice(size_t dim, size_t start, size_t len) const {
    return this->sliceImpl({{dim, start, len}});
}

std::shared_ptr<Tensor> Tensor::slice(const std::vector<SliceParams> &slices) {
    return this->sliceImpl(slices);
}

std::shared_ptr<Tensor const> Tensor::slice(const std::vector<SliceParams> &slices) const {
    return this->sliceImpl(slices);
}

void TensorDesc::dimMerge(size_t dim_start, size_t dim_end) {
    ASSERT(dim_start <= dim_end && dim_end < this->_shape.size());
    if (dim_start == dim_end) {
        return;
    }

    auto new_shape = std::vector<size_t>();
    auto new_strides = std::vector<ptrdiff_t>();
    for (size_t i = 0; i < dim_start; i++) {
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    for (size_t i = dim_start + 1; i <= dim_end; i++) {
        ASSERT_EQ(this->_strides[i - 1], ptrdiff_t(this->_shape[i]) * this->_strides[i]);
    }
    new_shape.push_back(std::accumulate(this->_shape.begin() + dim_start, this->_shape.begin() + dim_end + 1, 1, std::multiplies<size_t>()));
    new_strides.push_back(this->_strides[dim_end]);
    for (size_t i = dim_end + 1; i < this->_shape.size(); i++) {
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    this->_shape = new_shape;
    this->_strides = new_strides;
    this->resetDesc();
}

std::shared_ptr<Tensor> Tensor::dimMerge(size_t dim_start, size_t dim_end) {
    this->_desc->dimMerge(dim_start, dim_end);
    return shared_from_this();
}

// 在 dimSplit 函数中添加详细的错误调试信息

void TensorDesc::dimSplit(size_t dim, const std::vector<size_t> &dims) {
    // 计算 dims 向量中所有元素的乘积
    size_t dims_product = std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
    
    // 如果断言即将失败，输出详细的调试信息
    if (this->_shape[dim] != dims_product) {
        std::cerr << "=== dimSplit Error Debug Information ===" << std::endl;
        std::cerr << "Trying to split dimension " << dim << std::endl;
        std::cerr << "Original tensor shape: [";
        for (size_t i = 0; i < this->_shape.size(); i++) {
            std::cerr << this->_shape[i];
            if (i < this->_shape.size() - 1) std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        
        std::cerr << "Dimension " << dim << " current size: " << this->_shape[dim] << std::endl;
        
        std::cerr << "Requested split dimensions: [";
        for (size_t i = 0; i < dims.size(); i++) {
            std::cerr << dims[i];
            if (i < dims.size() - 1) std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        
        std::cerr << "Product of requested dimensions: " << dims_product << std::endl;
        std::cerr << "Expected: " << this->_shape[dim] << " == " << dims_product << std::endl;
        std::cerr << "Difference: " << (int64_t(this->_shape[dim]) - int64_t(dims_product)) << std::endl;
        std::cerr << "=========================================" << std::endl;
    }
    
    ASSERT_EQ(this->_shape[dim], dims_product);
    
    auto new_shape = std::vector<size_t>();
    auto new_strides = std::vector<ptrdiff_t>();
    for (size_t i = 0; i < dim; i++) {
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    for (size_t i = 0; i < dims.size(); i++) {
        new_shape.push_back(dims[i]);
        new_strides.push_back(this->_strides[dim] * this->_shape[dim] / std::accumulate(dims.begin(), dims.begin() + i + 1, 1, std::multiplies<size_t>()));
    }
    for (size_t i = dim + 1; i < this->_shape.size(); i++) {
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    this->_shape = new_shape;
    this->_strides = new_strides;
    this->resetDesc();
}

std::shared_ptr<Tensor> Tensor::dimSplit(size_t dim, const std::vector<size_t> &dims) {
    this->_desc->dimSplit(dim, dims);
    return shared_from_this();
}

void TensorDesc::permute(const std::vector<size_t> &order) {
    ASSERT_EQ(this->_shape.size(), order.size());
    auto new_shape = std::vector<size_t>(order.size());
    auto new_strides = std::vector<ptrdiff_t>(order.size());
    for (size_t i = 0; i < order.size(); i++) {
        ASSERT(std::find(order.begin(), order.end(), i) != order.end());
        new_shape[i] = this->_shape[order[i]];
        new_strides[i] = this->_strides[order[i]];
    }
    this->_shape = new_shape;
    this->_strides = new_strides;
    this->resetDesc();
}

std::shared_ptr<Tensor> Tensor::permute(const std::vector<size_t> &order) {
    this->_desc->permute(order);
    return shared_from_this();
}
