#include "runtime/tensor.hpp"
#include "../utils.hpp"
#include "execution/submit.hpp"
#include "runtime/context.hpp"
#include "runtime/dtype.hpp"
#include "runtime/runtime.hpp"

#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace {
// Helper function to calculate contiguous strides
inline infinilm::runtime::Strides calculate_contiguous_strides(const infinilm::runtime::Shape &shape) {
    infinilm::runtime::Strides strides(shape.size());
    size_t stride = 1;
    const auto max_stride = static_cast<size_t>(std::numeric_limits<infinilm::runtime::Stride>::max());
    for (size_t i = shape.size(); i-- > 0;) {
        if (stride > max_stride || (shape[i] != 0 && stride > max_stride / shape[i])) {
            throw std::length_error("tensor storage stride overflows");
        }
        strides[i] = static_cast<infinilm::runtime::Stride>(stride);
        stride *= shape[i];
    }
    return strides;
}

std::shared_ptr<infinilm::runtime::Memory> allocate_tensor_storage(
    infinilm::runtime::Runtime &runtime, size_t bytes, bool pinned) {
    return pinned ? runtime.allocatePinnedHostMemory(bytes) : runtime.allocateMemory(bytes);
}

size_t parameter_storage_bytes(const infinilm::runtime::Shape &shape,
                               const infinilm::runtime::DataType &dtype) {
    // Validate the physical Parameter layout before metadata or device binding.
    // A zero dimension is legal, but a huge suffix to its right is not.
    size_t elements = 1;
    const auto max_stride = static_cast<size_t>(std::numeric_limits<infinilm::runtime::Stride>::max());
    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        if (*it != 0 && elements > max_stride / *it) {
            throw std::length_error("parameter storage stride overflows");
        }
        elements *= *it;
    }
    if (elements == 0) {
        return 0;
    }
    const size_t element_bytes = infinilm::runtime::dsize(dtype);
    if (element_bytes == 0 || elements > std::numeric_limits<size_t>::max() / element_bytes) {
        throw std::length_error("parameter storage byte size overflows");
    }
    return elements * element_bytes;
}
} // namespace

namespace infinilm::runtime {
TensorImpl *Tensor::operator->() { return impl_.get(); }

const TensorImpl *Tensor::operator->() const { return impl_.get(); }

Tensor Tensor::empty(Runtime &runtime, const Shape &shape, const DataType &dtype, bool pin_memory) {
    return strided_empty(runtime, shape, calculate_contiguous_strides(shape), dtype, pin_memory);
}

Tensor Tensor::empty(const Shape &shape,
                     const DataType &dtype,
                     const Device &device,
                     bool pin_memory) {
    const bool pinned_host = pin_memory && device.type() == Device::Type::kCpu;
    auto runtime = pinned_host ? context::currentRuntime() : context::resolveRuntime(device);
    return empty(*runtime, shape, dtype, pinned_host);
}

Tensor Tensor::empty_parameter(Runtime &runtime,
                               const Shape &shape,
                               const DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.device().type() == Device::Type::kCpu) {
        return empty(runtime, shape, dtype);
    }
    const size_t bytes = parameter_storage_bytes(shape, dtype);
    if (bytes == 0) {
        return empty(runtime, shape, dtype);
    }
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, dtype));
    t->data_.offset = 0;
    t->data_.memory = runtime.allocateParameterMemory(bytes);
    return Tensor{t};
}

Tensor Tensor::strided_empty(Runtime &runtime,
                             const Shape &shape,
                             const Strides &strides,
                             const DataType &dtype, bool pin_memory) {
    runtime.assertOwnerThread();
    const size_t element_bytes = dsize(dtype);
    if (element_bytes == 0) {
        throw std::length_error("tensor element size is zero");
    }
    const size_t required_bytes = TensorImpl::storage_span_bytes(shape, strides, element_bytes);
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, strides, dtype));
    t->data_.offset = 0;
    t->data_.memory = allocate_tensor_storage(runtime, required_bytes, pin_memory);
    return Tensor{t};
}

Tensor Tensor::ones(const Shape &shape,
                    const DataType &dtype,
                    const Device &device,
                    bool pin_memory) {
    auto result = Tensor::empty(shape, dtype, device, pin_memory).impl_;
    if (result->nbytes() != 0) {
        auto runtime = context::resolveRuntime(result->device());
        infinilm::execution::submit<infini::ops::Fill>(
            *runtime, infini::ops::Fill::DefaultConfig(runtime->device().type()),
            Tensor{result}, 1.0, Tensor{result});
        INFINILM_TRACE_OP(Ones, Tensor{result});
    }
    return Tensor{result};
}

Tensor Tensor::from_blob(void *raw_ptr, const Shape &shape, const DataType &dtype, const Device &device) {
    return from_blob(raw_ptr, shape, dtype, device, nullptr);
}

Tensor Tensor::from_blob(void *raw_ptr,
                         const Shape &shape,
                         const DataType &dtype,
                         const Device &device,
                         std::shared_ptr<void> owner) {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, dtype));
    t->data_.offset = 0;
    const size_t size = t->nbytes();
    if (raw_ptr == nullptr && size != 0) {
        throw std::invalid_argument("from_blob requires a non-null pointer for non-empty tensors");
    }
    Memory::Deleter keep_owner;
    if (owner) {
        keep_owner = [owner = std::move(owner)](std::byte *) {};
    }
    t->data_.memory = std::make_shared<Memory>(
        static_cast<std::byte *>(raw_ptr), size, device, std::move(keep_owner));
    return Tensor{t};
}

Tensor Tensor::strided_from_blob(void *raw_ptr, const Shape &shape, const Strides &strides, const DataType &dtype, const Device &device) {
    return strided_from_blob(raw_ptr, shape, strides, dtype, device, nullptr);
}

Tensor Tensor::strided_from_blob(void *raw_ptr,
                                 const Shape &shape,
                                 const Strides &strides,
                                 const DataType &dtype,
                                 const Device &device,
                                 std::shared_ptr<void> owner) {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, strides, dtype));
    t->data_.offset = 0;
    const size_t size = TensorImpl::storage_span_bytes(shape, strides, dsize(dtype));
    if (raw_ptr == nullptr && size != 0) {
        throw std::invalid_argument(
            "strided_from_blob requires a non-null pointer for non-empty tensors");
    }
    Memory::Deleter keep_owner;
    if (owner) {
        keep_owner = [owner = std::move(owner)](std::byte *) {};
    }
    t->data_.memory = std::make_shared<Memory>(
        static_cast<std::byte *>(raw_ptr), size, device, std::move(keep_owner));
    return Tensor{t};
}

Tensor::operator bool() const {
    return impl_ != nullptr;
}

TensorMetaData::TensorMetaData(const Shape &_shape, const Strides &_strides, const DataType &_dtype)
    : shape(_shape), strides(_strides), dtype(_dtype) {}

TensorImpl::TensorImpl(const Shape &shape, const DataType &dtype)
    : meta_(TensorMetaData(shape, calculate_contiguous_strides(shape), dtype)) {}

TensorImpl::TensorImpl(const Shape &shape, const Strides &strides, const DataType &dtype)
    : meta_(TensorMetaData(shape, strides, dtype)) {}

std::byte *TensorImpl::data() {
    return data_.memory->data() + data_.offset;
}

const std::byte *TensorImpl::data() const {
    return data_.memory->data() + data_.offset;
}

const Shape &TensorImpl::shape() const {
    return meta_.shape;
}

const Strides &TensorImpl::strides() const {
    return meta_.strides;
}

Size TensorImpl::ndim() const {
    return meta_.shape.size();
}

bool TensorImpl::is_contiguous() const {
    Stride expected_stride = 1;
    for (int i = meta_.shape.size() - 1; i >= 0; --i) {
        if (meta_.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= meta_.shape[i];
    }
    return true;
}

Size TensorImpl::numel() const {
    Size total = 1;
    for (const auto &dim : meta_.shape) {
        total *= dim;
    }
    return total;
}

size_t TensorImpl::element_size() const {
    return dsize(dtype());
}

size_t TensorImpl::nbytes() const {
    return numel() * element_size();
}

Size TensorImpl::size(size_t dim) const {
    return meta_.shape[dim];
}

Stride TensorImpl::stride(size_t dim) const {
    return meta_.strides[dim];
}

DataType TensorImpl::dtype() const {
    return meta_.dtype;
}

Device TensorImpl::device() const {
    return data_.memory->device();
}

infini::rt::TensorView TensorImpl::view() {
    return infini::rt::TensorView{data(), shape(), dtype(), device(), strides()};
}

bool TensorImpl::is_pinned() const {
    return data_.memory->is_pinned();
}

std::string TensorImpl::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << toString(this->dtype());
    ss << " device=" << this->device().ToString();

    return ss.str();
}

size_t TensorImpl::storage_span_bytes(
    const Shape &shape,
    const Strides &strides,
    size_t element_size) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument("tensor shape and strides must have the same rank");
    }
    if (std::any_of(shape.begin(), shape.end(), [](Size dim) { return dim == 0; })) {
        return 0;
    }

    size_t max_offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (strides[i] < 0) {
            throw std::invalid_argument("negative tensor strides are not supported");
        }
        const auto stride = static_cast<size_t>(strides[i]);
        const auto extent = shape[i] - 1;
        if (extent != 0 && stride > std::numeric_limits<size_t>::max() / extent) {
            throw std::overflow_error("tensor storage span overflows size_t");
        }
        const auto contribution = extent * stride;
        if (contribution > std::numeric_limits<size_t>::max() - max_offset) {
            throw std::overflow_error("tensor storage span overflows size_t");
        }
        max_offset += contribution;
    }
    if (max_offset == std::numeric_limits<size_t>::max()
        || max_offset + 1 > std::numeric_limits<size_t>::max() / element_size) {
        throw std::overflow_error("tensor storage size overflows size_t");
    }
    return (max_offset + 1) * element_size;
}

Tensor TensorImpl::to_blob_(Runtime &runtime) const {
    runtime.assertOwnerThread();
    const bool graph_pinnable = data_.memory->is_graph_pinnable();
    const bool keep_preexisting_owner = graph_pinnable
                                     && runtime.isGraphRecording() && runtime.retainGraphMemory(data_.memory);
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape(), strides(), dtype()));
    t->data_.offset = data_.offset;
    Memory::Deleter keep_external_owner;
    if (!graph_pinnable || keep_preexisting_owner) {
        keep_external_owner = [memory = data_.memory](std::byte *) {};
    }
    t->data_.memory = std::make_shared<Memory>(
        data_.memory->data(), data_.memory->size(), data_.memory->device(),
        std::move(keep_external_owner), false, false, data_.memory->allocation_owner());
    t->to_blob_mark_ = graph_pinnable;
    return Tensor{t};
}

Tensor TensorImpl::resume_from_blob_(Runtime &runtime) const {
    runtime.assertOwnerThread();
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape(), strides(), dtype()));
    t->data_.offset = data_.offset;
    t->data_.memory = to_blob_mark_
                        ? runtime.reinstantiateBlob(data_.memory)
                        : data_.memory;
    return Tensor{t};
}

} // namespace infinilm::runtime
