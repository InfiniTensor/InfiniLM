#include "../utils.h"
#include "tensor.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>

__C __export infiniStatus_t infiniopCreateTensorDescriptor(infiniopTensorDescriptor_t *desc_ptr, size_t ndim, size_t const *shape_, ptrdiff_t const *strides_, infiniDtype_t datatype) {
    if (strides_ != nullptr) {
        *desc_ptr = new InfiniopTensorDescriptor(datatype, ndim, shape_, strides_);
    } else {
        std::vector<ptrdiff_t> strides(ndim);
        ptrdiff_t dsize = 1;
        if (ndim > 0) {
            for (int i = (int)ndim - 1; i >= 0; i--) {
                strides[i] = dsize;
                dsize *= shape_[i];
            }
        }
        *desc_ptr = new InfiniopTensorDescriptor(datatype, ndim, shape_, strides.data());
    }

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopDestroyTensorDescriptor(infiniopTensorDescriptor_t desc) {
    delete desc;
    return INFINI_STATUS_SUCCESS;
}

InfiniopTensorDescriptor::InfiniopTensorDescriptor(infiniDtype_t dtype, size_t ndim, const size_t *shape, const ptrdiff_t *strides) {
    _dtype = dtype;
    _shape = std::vector<size_t>(shape, shape + ndim);
    _strides = std::vector<ptrdiff_t>(strides, strides + ndim);
}

infiniDtype_t InfiniopTensorDescriptor::dtype() const {
    return _dtype;
}

std::vector<size_t> InfiniopTensorDescriptor::shape() const {
    return std::vector<size_t>(_shape.begin(), _shape.end());
}

size_t InfiniopTensorDescriptor::dim(size_t i) const {
    return _shape[i];
}

size_t InfiniopTensorDescriptor::ndim() const {
    return _shape.size();
}

std::vector<ptrdiff_t> InfiniopTensorDescriptor::strides() const {
    return std::vector<ptrdiff_t>(_strides.begin(), _strides.end());
}

ptrdiff_t InfiniopTensorDescriptor::stride(size_t i) const {
    return _strides[i];
}

size_t InfiniopTensorDescriptor::numel() const {
    return std::accumulate(_shape.begin(), _shape.end(), (size_t)1, std::multiplies<size_t>());
}

std::vector<ptrdiff_t> InfiniopTensorDescriptor::getByteStrides() const {
    std::vector<ptrdiff_t> byte_strides(_shape.size());
    for (size_t i = 0; i < _shape.size(); i++) {
        byte_strides[i] = _strides[i] * infiniSizeOf(_dtype);
    }
    return byte_strides;
}

bool InfiniopTensorDescriptor::isContiguous(size_t dim_start, size_t dim_end) const {
    if (ndim() == 0) {
        return true;
    }
    for (size_t i = dim_start + 1; i <= dim_end; i++) {
        if (stride(i - 1) != static_cast<ptrdiff_t>(dim(i)) * stride(i)) {
            return false;
        }
    }
    return true;
}

bool InfiniopTensorDescriptor::isContiguous() const {
    return isContiguous(0, ndim() - 1);
}

bool InfiniopTensorDescriptor::hasBroadcastDim() const {
    return std::any_of(
        _shape.begin(), _shape.end(),
        [&, i = 0](const auto &) mutable {
            return _shape[i] != 1 && _strides[i++] == 0;
        });
}

std::vector<size_t> InfiniopTensorDescriptor::getBroadcastDim() const {
    std::vector<size_t> res;
    for (size_t i = 0; i < ndim(); ++i) {
        if (_shape[i] != 1 && _strides[i] == 0) {
            res.push_back(i);
        }
    }
    return res;
}

utils::Result<infiniopTensorDescriptor_t> InfiniopTensorDescriptor::dimMerge(size_t dim_start, size_t dim_end) const {
    CHECK_OR_RETURN(dim_start <= dim_end && dim_end < ndim(), INFINI_STATUS_BAD_PARAM);

    size_t new_ndim = ndim() - (dim_end - dim_start);
    std::vector<size_t> new_shape(new_ndim);
    std::vector<ptrdiff_t> new_strides(new_ndim);
    size_t index = 0;

    for (size_t i = 0; i < dim_start; i++) {
        new_shape[index] = dim(i);
        new_strides[index] = stride(i);
        index++;
    }

    CHECK_OR_RETURN(isContiguous(dim_start, dim_end), INFINI_STATUS_BAD_PARAM);

    new_shape[index] = 1;
    for (size_t i = dim_start; i <= dim_end; i++) {
        new_shape[index] *= dim(i);
    }

    new_strides[index] = stride(dim_end);
    index++;

    for (size_t i = dim_end + 1; i < ndim(); i++) {
        new_shape[index] = dim(i);
        new_strides[index] = stride(i);
        index++;
    }

    return utils::Result<infiniopTensorDescriptor_t>(
        new InfiniopTensorDescriptor(_dtype, new_ndim, new_shape.data(), new_strides.data()));
}

utils::Result<infiniopTensorDescriptor_t> InfiniopTensorDescriptor::dimSplit(size_t axis, const std::vector<size_t> &dims) const {
    size_t ndim_ = ndim();

    CHECK_OR_RETURN(dim(axis) == std::accumulate(dims.begin(), dims.end(), (size_t)1, std::multiplies<size_t>()),
                    INFINI_STATUS_BAD_PARAM);

    size_t new_ndim = ndim_ + dims.size() - 1;
    std::vector<size_t> new_shape(new_ndim);
    std::vector<ptrdiff_t> new_strides(new_ndim);
    size_t index = 0;
    for (size_t i = 0; i < axis; i++) {
        new_shape[index] = dim(i);
        new_strides[index] = stride(i);
        index++;
    }
    for (size_t i = 0; i < dims.size(); i++) {
        new_shape[index] = dims[i];
        new_strides[index] = stride(axis) * dim(axis) / std::accumulate(dims.begin(), dims.begin() + i + 1, (size_t)1, std::multiplies<size_t>());
        index++;
    }
    for (size_t i = axis + 1; i < ndim_; i++) {
        new_shape[index] = dim(i);
        new_strides[index] = stride(i);
        index++;
    }

    return utils::Result<infiniopTensorDescriptor_t>(
        new InfiniopTensorDescriptor(_dtype, new_ndim, new_shape.data(), new_strides.data()));
}

utils::Result<infiniopTensorDescriptor_t> InfiniopTensorDescriptor::dimPermute(const std::vector<size_t> &order) const {
    auto ndim_ = ndim();
    CHECK_OR_RETURN(order.size() == ndim_, INFINI_STATUS_BAD_PARAM);
    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        CHECK_OR_RETURN(std::find(order.begin(), order.end(), i) != order.end(), INFINI_STATUS_BAD_PARAM);
        new_shape[i] = dim(order[i]);
        new_strides[i] = stride(order[i]);
    }
    return utils::Result<infiniopTensorDescriptor_t>(
        new InfiniopTensorDescriptor(_dtype, ndim_, new_shape.data(), new_strides.data()));
}

std::string InfiniopTensorDescriptor::toString() const {
    std::string str = "dtype: " + infiniDtypeToString(_dtype) + ", shape: [";
    for (size_t i = 0; i < ndim(); i++) {
        str += std::to_string(dim(i)) + (i == ndim() - 1 ? "" : ", ");
    }
    str += "], strides: [";
    for (size_t i = 0; i < ndim(); i++) {
        str += std::to_string(stride(i)) + (i == ndim() - 1 ? "" : ", ");
    }
    str += "]";
    return str;
}
