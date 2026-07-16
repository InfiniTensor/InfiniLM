#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include "../utils.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinicore {
Tensor TensorImpl::squeeze(size_t dim) const {
    // Create new shape with dimension of size one removed at dim
    if (meta_.shape[dim] != 1) {
        spdlog::error("Dimension {} is not of size 1 for squeeze operation on {}.", dim, this->info());
        throw std::runtime_error("Invalid squeeze operation on tensor.");
    }
    Shape new_shape = meta_.shape;
    new_shape.erase(new_shape.begin() + dim);
    Strides new_strides = meta_.strides;
    new_strides.erase(new_strides.begin() + dim);

    auto tensor_impl = std::make_shared<TensorImpl>(new_shape, new_strides, meta_.dtype);
    tensor_impl->data_ = data_;

    return Tensor(tensor_impl);
}

Tensor TensorImpl::unsqueeze(size_t dim) const {
    // Create new shape with dimension of size one inserted at dim
    Shape new_shape = meta_.shape;
    new_shape.insert(new_shape.begin() + dim, 1);

    // Create new strides with stride of zero for the new dimension
    Strides new_strides = meta_.strides;
    new_strides.insert(new_strides.begin() + dim, 0);

    auto tensor_impl = std::make_shared<TensorImpl>(new_shape, new_strides, meta_.dtype);
    tensor_impl->data_ = data_;

    return Tensor(tensor_impl);
}

Tensor TensorImpl::narrow(const std::vector<TensorSliceParams> &slices) const {
    // Create new shape and calculate offset
    Shape new_shape = meta_.shape;
    size_t offset = data_.offset;

    for (const auto &slice : slices) {
        if (meta_.shape[slice.dim] < slice.start + slice.len) {
            spdlog::error("Invalid slice [dim={}, start={}, len={}] on {}.", slice.dim, slice.start, slice.len, this->info());
            throw std::runtime_error("Invalid slice on tensor.");
        }
        new_shape[slice.dim] = slice.len;
        offset += slice.start * meta_.strides[slice.dim] * dsize(meta_.dtype);
    }

    // Create new tensor with the same strides but narrowed shape
    auto tensor_impl = std::make_shared<TensorImpl>(new_shape, meta_.strides, meta_.dtype);
    tensor_impl->data_.offset = offset;
    tensor_impl->data_.memory = data_.memory;

    return Tensor(tensor_impl);
}

Tensor TensorImpl::permute(const Shape &order) const {
    // Validate input
    INFINICORE_ASSERT(meta_.shape.size() == order.size());

    // Check that order contains all indices from 0 to n-1 exactly once
    for (size_t i = 0; i < order.size(); i++) {
        INFINICORE_ASSERT(std::find(order.begin(), order.end(), i) != order.end());
    }

    // Permute shape and strides
    Shape new_shape(order.size());
    Strides new_strides(order.size());

    for (size_t i = 0; i < order.size(); i++) {
        new_shape[i] = meta_.shape[order[i]];
        new_strides[i] = meta_.strides[order[i]];
    }

    auto tensor_impl = std::make_shared<TensorImpl>(new_shape, new_strides, meta_.dtype);
    tensor_impl->data_ = data_;

    return Tensor(tensor_impl);
}

Tensor TensorImpl::view(const Shape &new_shape) const {
    // Step 1: Validate total size
    Size numel = 1;
    for (Size dim : meta_.shape) {
        numel *= dim;
    }

    Size new_numel = 1;
    for (Size dim : new_shape) {
        new_numel *= dim;
    }

    assert(numel == new_numel);

    // Step 2: Get current shape and strides
    const Shape &old_shape = meta_.shape;
    const Strides &old_strides = meta_.strides;

    // Step 3: Create merged shape and strides
    Shape merged_shape;
    Strides merged_strides;

    if (!old_shape.empty()) {
        merged_shape.push_back(old_shape[0]);
        merged_strides.push_back(old_strides[0]);

        for (size_t i = 1; i < old_shape.size(); ++i) {
            if (old_strides[i] * static_cast<Stride>(old_shape[i]) == merged_strides.back()) {
                merged_shape.back() *= old_shape[i];
                merged_strides.back() = old_strides[i];
            } else {
                merged_shape.push_back(old_shape[i]);
                merged_strides.push_back(old_strides[i]);
            }
        }
    }

    // Step 4: Compute new strides by splitting merged dimensions
    Strides new_strides(new_shape.size());
    size_t merged_idx = 0;
    Stride current_stride = merged_strides[0];
    Size remaining_size = merged_shape[0];

    for (size_t i = 0; i < new_shape.size(); ++i) {
        // Find which merged dimension contains this new dimension
        while (new_shape[i] > remaining_size) {
            if (++merged_idx >= merged_shape.size()) {
                throw std::runtime_error("Incompatible shape for view operation.");
            }
            current_stride = merged_strides[merged_idx];
            remaining_size = merged_shape[merged_idx];
        }

        if (remaining_size % new_shape[i] != 0) {
            throw std::runtime_error("Incompatible shape for view operation.");
        };

        new_strides[i] = current_stride * (remaining_size / new_shape[i]);
        remaining_size /= new_shape[i];
    }

    return this->as_strided(new_shape, new_strides);
}

Tensor TensorImpl::as_strided(const Shape &new_shape, const Strides &new_strides) const {
    auto tensor_impl = std::make_shared<TensorImpl>(new_shape, new_strides, meta_.dtype);
    tensor_impl->data_ = data_;

    return Tensor(tensor_impl);
}
} // namespace infinicore
