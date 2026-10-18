#include "runtime/context.hpp"
#include "runtime/dtype.hpp"
#include "runtime/tensor.hpp"

#include "../utils.hpp"

#include <limits>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>

namespace infinilm::runtime {
namespace {
Size checked_view_numel(const Shape &shape) {
    if (std::find(shape.begin(), shape.end(), Size{0}) != shape.end()) {
        return 0;
    }
    Size result = 1;
    for (Size dim : shape) {
        if (result > std::numeric_limits<Size>::max() / dim) {
            throw std::length_error("view element count overflows");
        }
        result *= dim;
    }
    return result;
}

Stride checked_view_stride(Stride stride, Size factor) {
    if (stride < 0 || (stride != 0 && factor > static_cast<Size>(std::numeric_limits<Stride>::max() / stride))) {
        throw std::length_error("view stride overflows");
    }
    return stride == 0 ? 0 : stride * static_cast<Stride>(factor);
}

std::string shape_to_string(const Shape &shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string strides_to_string(const Strides &strides) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << strides[i];
    }
    oss << "]";
    return oss.str();
}

std::string incompatible_view_message(
    const Shape &old_shape,
    const Strides &old_strides,
    const Shape &new_shape) {
    return "Incompatible shape for view operation: old_shape="
         + shape_to_string(old_shape)
         + " old_strides=" + strides_to_string(old_strides)
         + " new_shape=" + shape_to_string(new_shape);
}
} // namespace

Tensor TensorImpl::squeeze(size_t dim) const {
    if (dim >= meta_.shape.size()) {
        throw std::out_of_range("squeeze dimension is out of range");
    }
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
    if (dim > meta_.shape.size()) {
        throw std::out_of_range("unsqueeze dimension is out of range");
    }
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
        if (slice.dim >= meta_.shape.size()
            || slice.start > meta_.shape[slice.dim]
            || slice.len > meta_.shape[slice.dim] - slice.start) {
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
    INFINILM_ASSERT(meta_.shape.size() == order.size());

    // Check that order contains all indices from 0 to n-1 exactly once
    for (size_t i = 0; i < order.size(); i++) {
        INFINILM_ASSERT(std::find(order.begin(), order.end(), i) != order.end());
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
    const Size numel = checked_view_numel(meta_.shape);
    const Size new_numel = checked_view_numel(new_shape);
    if (numel != new_numel) {
        throw std::invalid_argument(
            incompatible_view_message(meta_.shape, meta_.strides, new_shape));
    }

    // Empty tensors have no dimensions to split; scalars have no merged
    // dimensions. Both still share the original Memory and byte offset.
    if (numel == 0 || meta_.shape.empty()) {
        if (new_shape == meta_.shape) {
            return this->as_strided(new_shape, meta_.strides);
        }
        Strides new_strides(new_shape.size());
        Stride stride = 1;
        for (size_t i = new_shape.size(); i-- > 0;) {
            new_strides[i] = stride;
            stride = checked_view_stride(stride, std::max(Size{1}, new_shape[i]));
        }
        return this->as_strided(new_shape, new_strides);
    }

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
            if (checked_view_stride(old_strides[i], old_shape[i]) == merged_strides.back()) {
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
                throw std::runtime_error(
                    incompatible_view_message(old_shape, old_strides, new_shape));
            }
            current_stride = merged_strides[merged_idx];
            remaining_size = merged_shape[merged_idx];
        }

        if (remaining_size % new_shape[i] != 0) {
            throw std::runtime_error(
                incompatible_view_message(old_shape, old_strides, new_shape));
        };

        new_strides[i] = checked_view_stride(current_stride, remaining_size / new_shape[i]);
        remaining_size /= new_shape[i];
    }

    return this->as_strided(new_shape, new_strides);
}

Tensor TensorImpl::as_strided(const Shape &new_shape, const Strides &new_strides) const {
    const auto required_bytes = storage_span_bytes(
        new_shape, new_strides, dsize(meta_.dtype));
    if (data_.offset > data_.memory->size()
        || required_bytes > data_.memory->size() - data_.offset) {
        std::ostringstream message;
        message << "as_strided view exceeds the underlying storage: shape="
                << shape_to_string(shape()) << " strides=" << strides_to_string(strides())
                << " requested_shape=" << shape_to_string(new_shape)
                << " requested_strides=" << strides_to_string(new_strides)
                << " offset=" << data_.offset << " span=" << required_bytes
                << " storage=" << data_.memory->size();
        throw std::out_of_range(message.str());
    }
    auto tensor_impl = std::make_shared<TensorImpl>(new_shape, new_strides, meta_.dtype);
    tensor_impl->data_ = data_;

    return Tensor(tensor_impl);
}
} // namespace infinilm::runtime
