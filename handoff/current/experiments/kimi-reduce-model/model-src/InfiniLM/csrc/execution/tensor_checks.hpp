#pragma once
#include "submit.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
namespace infinilm::execution::tensor_checks {
using namespace infinilm::runtime;
namespace pad {

struct PadLayout {
    Shape output_shape;
};

inline PadLayout analyze(const Tensor &x, const std::vector<int> &padding,
                         const std::string &mode) {
    if (!x) {
        throw std::invalid_argument("Pad input is null");
    }
    // The frozen LM consumers all use constant mode. The old generic API also
    // named reflect/replicate/circular; none is implemented by PadConstant.
    if (mode != "constant") {
        throw std::invalid_argument("Pad mode requires an unimplemented nonconstant provider");
    }
    if (padding.empty() || padding.size() % 2 != 0 || padding.size() / 2 > x->ndim()) {
        throw std::invalid_argument("Pad requires pairs for at most the input rank");
    }

    PadLayout layout{x->shape()};
    for (size_t pair = 0; pair < padding.size() / 2; ++pair) {
        const int left = padding[2 * pair];
        const int right = padding[2 * pair + 1];
        if (left < 0 || right < 0) {
            throw std::invalid_argument("Pad does not crop negative padding");
        }
        const size_t axis = x->ndim() - 1 - pair;
        const size_t left_size = static_cast<size_t>(left);
        const size_t right_size = static_cast<size_t>(right);
        if (left_size > std::numeric_limits<size_t>::max() - layout.output_shape[axis] || right_size > std::numeric_limits<size_t>::max() - (layout.output_shape[axis] + left_size)) {
            throw std::overflow_error("Pad output dimension overflows");
        }
        layout.output_shape[axis] += left_size + right_size;
    }

    size_t element_count = 1;
    size_t stride_product = 1;
    for (const size_t extent : layout.output_shape) {
        if (extent && stride_product > static_cast<size_t>(std::numeric_limits<Stride>::max()) / extent) {
            throw std::overflow_error("Pad output strides overflow");
        }
        if (extent) {
            stride_product *= extent;
        }
        element_count *= extent;
    }
    if (element_count && x->element_size() > std::numeric_limits<size_t>::max() / element_count) {
        throw std::overflow_error("Pad output bytes overflow");
    }
    return layout;
}

} // namespace pad
namespace take {
inline void validate(Runtime &runtime, const Tensor &output, const Tensor &input,
                     const Tensor &indices) {
    runtime.assertOwnerThread();
    if (!output || !input || !indices) {
        throw std::invalid_argument("Take requires non-null tensors");
    }
    if (output->device() != runtime.device() || input->device() != runtime.device()
        || indices->device() != runtime.device()) {
        throw std::invalid_argument("Take runtime and tensor devices differ");
    }
    if (indices->dtype() != DataType::kInt64 || output->dtype() != input->dtype()
        || output->shape() != indices->shape()) {
        throw std::invalid_argument("Take index or output metadata is invalid");
    }
}
} // namespace take
namespace sum {
inline std::vector<size_t> normalized_dim(const Tensor &input,
                                          std::vector<size_t> dim) {
    if (dim.empty()) {
        for (size_t axis = 0; axis < input->ndim(); ++axis) {
            dim.push_back(axis);
        }
    }
    std::sort(dim.begin(), dim.end());
    if (std::adjacent_find(dim.begin(), dim.end()) != dim.end()
        || (!dim.empty() && dim.back() >= input->ndim())) {
        throw std::invalid_argument("Sum has duplicate or out-of-range dimensions");
    }
    return dim;
}

inline Shape reduced_shape(const Tensor &input, const std::vector<size_t> &dim,
                           bool keepdim) {
    Shape output;
    output.reserve(input->ndim());
    size_t reduced = 0;
    for (size_t axis = 0; axis < input->ndim(); ++axis) {
        if (reduced < dim.size() && dim[reduced] == axis) {
            if (keepdim) {
                output.push_back(1);
            }
            ++reduced;
        } else {
            output.push_back(input->size(axis));
        }
    }
    return output;
}

inline void validate(Runtime &runtime, const Tensor &output, const Tensor &input,
                     const std::vector<size_t> &dim, bool keepdim) {
    runtime.assertOwnerThread();
    if (!output || !input || output->device() != runtime.device()
        || input->device() != runtime.device()) {
        throw std::invalid_argument("Sum requires tensors on its Runtime device");
    }
    if (output->dtype() != input->dtype()
        || output->shape() != reduced_shape(input, dim, keepdim)) {
        throw std::invalid_argument("Sum output shape or dtype differs from reduction");
    }
}
} // namespace sum
namespace broadcast_to {

inline void validate_shape(const Shape &source, const Shape &target) {
    if (source.size() > target.size()) {
        throw std::invalid_argument("BroadcastTo output rank is smaller than input rank");
    }
    for (size_t i = 0; i < source.size(); ++i) {
        const size_t from = source[source.size() - 1 - i];
        const size_t to = target[target.size() - 1 - i];
        if (from != 1 && from != to) {
            throw std::invalid_argument("BroadcastTo dimensions are not right-aligned broadcastable");
        }
    }
}

inline Shape checked_shape(const std::vector<int64_t> &shape, size_t element_size) {
    Shape target;
    target.reserve(shape.size());
    size_t count = 1;
    size_t layout_product = 1;
    for (const auto dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("BroadcastTo output dimensions must be nonnegative");
        }
        const auto extent = static_cast<size_t>(dim);
        // Tensor::empty calculates signed contiguous strides even for an
        // empty output, so zero axes cannot mask another dimension's overflow.
        if (extent && layout_product > static_cast<size_t>(std::numeric_limits<Stride>::max()) / extent) {
            throw std::overflow_error("BroadcastTo output strides overflow");
        }
        if (extent) {
            layout_product *= extent;
        }
        count *= extent;
        target.push_back(extent);
    }
    if (count && element_size > std::numeric_limits<size_t>::max() / count) {
        throw std::overflow_error("BroadcastTo output bytes overflow");
    }
    return target;
}
} // namespace broadcast_to
namespace cat {
inline bool empty_vector(const Tensor &tensor) {
    const auto &shape = tensor->shape();
    return shape.size() == 1 && shape[0] == 0;
}

struct Layout {
    const Tensor *reference;
    size_t axis;
    size_t length;
};

inline Layout analyze(Runtime &runtime, const std::vector<Tensor> &inputs, int dim) {
    runtime.assertOwnerThread();
    if (inputs.empty()) {
        throw std::invalid_argument("Cat requires input tensors");
    }
    const Tensor *reference = nullptr;
    const auto device = runtime.device();
    for (const auto &input : inputs) {
        if (!input) {
            throw std::invalid_argument("Cat input is null");
        }
        reference = &input;
        if (!empty_vector(input)) {
            break;
        }
    }
    const auto *ref = reference->operator->();
    const auto &shape = ref->shape();
    const auto dtype = ref->dtype();
    const auto rank = shape.size();
    if (rank == 0 || dim < -static_cast<int64_t>(rank) || dim >= static_cast<int64_t>(rank)) {
        throw std::invalid_argument("Cat dimension is outside the input rank");
    }
    Layout layout{reference,
                  static_cast<size_t>(dim < 0 ? dim + static_cast<int64_t>(rank) : dim),
                  0};
    for (const auto &input : inputs) {
        if (!input) {
            throw std::invalid_argument("Cat input is null");
        }
        const auto *tensor = input.operator->();
        if (tensor->device() != device) {
            throw std::invalid_argument("Cat input must belong to its Runtime");
        }
        const auto &dims = tensor->shape();
        if (tensor->dtype() != dtype) {
            throw std::invalid_argument("Cat input dtypes differ");
        }
        if (dims.size() == 1 && dims[0] == 0) {
            continue;
        }
        if (dims.size() != rank) {
            throw std::invalid_argument("Cat input ranks differ");
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            if (axis != layout.axis && dims[axis] != shape[axis]) {
                throw std::invalid_argument("Cat non-concatenated dimensions differ");
            }
        }
        if (dims[layout.axis] > std::numeric_limits<size_t>::max() - layout.length) {
            throw std::overflow_error("Cat concatenated dimension overflows");
        }
        layout.length += dims[layout.axis];
    }
    return layout;
}

inline void validate(Runtime &runtime, Tensor &out, const std::vector<Tensor> &inputs,
                     const Layout &layout) {
    runtime.assertOwnerThread();
    if (!out) {
        throw std::invalid_argument("Cat output is null");
    }
    if (inputs.empty()) {
        throw std::invalid_argument("Cat requires input tensors");
    }
    auto *dst = out.operator->();
    const auto &dims = dst->shape();
    if (dims.empty() || layout.axis >= dims.size() || dst->device() != runtime.device()) {
        throw std::invalid_argument("Cat output shape or device differs");
    }
    size_t length = 0;
    for (const auto &input : inputs) {
        if (!input) {
            throw std::invalid_argument("Cat input is null");
        }
        const auto *src = input.operator->();
        if (src->device() != runtime.device() || src->dtype() != dst->dtype()) {
            throw std::invalid_argument("Cat input dtype or device differs");
        }
        if (empty_vector(input)) {
            continue;
        }
        const auto &shape = src->shape();
        if (shape.size() != dims.size()) {
            throw std::invalid_argument("Cat input ranks differ");
        }
        for (size_t axis = 0; axis < dims.size(); ++axis) {
            if (axis != layout.axis && shape[axis] != dims[axis]) {
                throw std::invalid_argument("Cat non-concatenated dimensions differ");
            }
        }
        if (shape[layout.axis] > std::numeric_limits<size_t>::max() - length) {
            throw std::overflow_error("Cat concatenated dimension overflows");
        }
        length += shape[layout.axis];
    }
    if (length != dims[layout.axis]) {
        throw std::invalid_argument("Cat output shape differs");
    }
}
} // namespace cat
} // namespace infinilm::execution::tensor_checks
