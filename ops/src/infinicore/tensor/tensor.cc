#include "infinicore/tensor.hpp"
#include "../context/internal.hpp"
#include "../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"

#include <spdlog/spdlog.h>

namespace {
// Helper function to calculate contiguous strides
inline infinicore::Strides calculate_contiguous_strides(const infinicore::Shape &shape) {
    infinicore::Strides strides(shape.size());
    infinicore::Stride stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}
} // namespace

namespace infinicore {
TensorImpl *Tensor::operator->() { return impl_.get(); }

const TensorImpl *Tensor::operator->() const { return impl_.get(); }

Tensor Tensor::empty(const Shape &shape,
                     const DataType &dtype,
                     const Device &device,
                     bool pin_memory) {
    return Tensor{TensorImpl::empty(shape, dtype, device, pin_memory)};
}

Tensor Tensor::strided_empty(const Shape &shape,
                             const Strides &strides,
                             const DataType &dtype,
                             const Device &device,
                             bool pin_memory) {
    return Tensor{TensorImpl::strided_empty(shape, strides, dtype, device, pin_memory)};
}

Tensor Tensor::zeros(const Shape &shape,
                     const DataType &dtype,
                     const Device &device,
                     bool pin_memory) {
    return Tensor{TensorImpl::zeros(shape, dtype, device, pin_memory)};
}

Tensor Tensor::ones(const Shape &shape,
                    const DataType &dtype,
                    const Device &device,
                    bool pin_memory) {
    return Tensor{TensorImpl::ones(shape, dtype, device, pin_memory)};
}

Tensor Tensor::from_blob(void *raw_ptr, const Shape &shape, const DataType &dtype, const Device &device) {
    return Tensor{TensorImpl::from_blob(raw_ptr, shape, dtype, device)};
}

Tensor Tensor::strided_from_blob(void *raw_ptr, const Shape &shape, const Strides &strides, const DataType &dtype, const Device &device) {
    return Tensor{TensorImpl::strided_from_blob(raw_ptr, shape, strides, dtype, device)};
}

Tensor::operator bool() const {
    return impl_ != nullptr;
}

TensorMetaData::TensorMetaData(const Shape &_shape, const Strides &_strides, const DataType &_dtype)
    : shape(_shape), strides(_strides), dtype(_dtype) {
    INFINICORE_CHECK_ERROR(infiniopCreateTensorDescriptor(&desc, shape.size(), shape.data(), strides.data(), (infiniDtype_t)dtype));
}

TensorMetaData::~TensorMetaData() {
    if (desc) {
        infiniopDestroyTensorDescriptor(desc);
        desc = nullptr;
    }
}

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

infiniopTensorDescriptor_t TensorImpl::desc() const {
    return meta_.desc;
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
    ss << " device=" << this->device().toString();

    return ss.str();
}

std::shared_ptr<TensorImpl> TensorImpl::empty(const Shape &shape,
                                              const DataType &dtype,
                                              const Device &device,
                                              bool pin_memory) {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, dtype));
    t->data_.offset = 0;

    if (device == Device::Type::CPU) {
        if (pin_memory) {
            if (context::getDevice() == Device::Type::CPU) {
                spdlog::warn("Tensor memory is not pinned by any device with CPU runtime.");
                t->data_.memory = context::allocateHostMemory(t->numel() * dsize(dtype));
            } else {
                t->data_.memory = context::allocatePinnedHostMemory(t->numel() * dsize(dtype));
            }
        } else {
            t->data_.memory = context::allocateHostMemory(t->numel() * dsize(dtype));
        }
    } else {
        context::setDevice(device);
        t->data_.memory = context::allocateMemory(t->numel() * dsize(dtype));
    }

    return t;
}

std::shared_ptr<TensorImpl> TensorImpl::strided_empty(
    const Shape &shape,
    const Strides &strides,
    const DataType &dtype,
    const Device &device,
    bool pin_memory) {

    auto impl = std::shared_ptr<TensorImpl>(new TensorImpl(shape, strides, dtype));
    impl->data_.offset = 0;

    size_t max_offset = 0;

    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > 0) {
            max_offset += (shape[i] - 1) * strides[i];
        }
    }

    size_t required_elements = max_offset + 1;
    size_t required_bytes = required_elements * dsize(dtype);

    if (device == Device::Type::CPU) {
        if (pin_memory) {
            if (context::getDevice() == Device::Type::CPU) {
                spdlog::warn("Tensor memory is not pinned by any device with CPU runtime.");
                impl->data_.memory = context::allocateHostMemory(required_bytes);
            } else {
                impl->data_.memory = context::allocatePinnedHostMemory(required_bytes);
            }
        } else {
            impl->data_.memory = context::allocateHostMemory(required_bytes);
        }
    } else {
        context::setDevice(device);
        impl->data_.memory = context::allocateMemory(required_bytes);
    }

    return impl;
}

std::shared_ptr<TensorImpl> TensorImpl::zeros(const Shape &shape,
                                              const DataType &dtype,
                                              const Device &device,
                                              bool pin_memory) {
    // TODO: Implement this.
    return empty(shape, dtype, device, pin_memory);
}
std::shared_ptr<TensorImpl> TensorImpl::ones(const Shape &shape,
                                             const DataType &dtype,
                                             const Device &device,
                                             bool pin_memory) {
    // TODO: Implement this.
    return empty(shape, dtype, device, pin_memory);
}

std::shared_ptr<TensorImpl> TensorImpl::from_blob(
    void *raw_ptr,
    const Shape &shape,
    const DataType &dtype,
    const Device &device) {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, dtype));
    t->data_.offset = 0;
    t->data_.memory = std::make_shared<Memory>((std::byte *)raw_ptr, t->numel() * dsize(dtype), device, nullptr);
    return t;
}

std::shared_ptr<TensorImpl> TensorImpl::strided_from_blob(
    void *raw_ptr,
    const Shape &shape,
    const Strides &strides,
    const DataType &dtype,
    const Device &device) {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, strides, dtype));
    t->data_.offset = 0;
    t->data_.memory = std::make_shared<Memory>((std::byte *)raw_ptr, t->numel() * dsize(dtype), device, nullptr);
    return t;
}

Tensor TensorImpl::to_blob_() const {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape(), strides(), dtype()));
    t->data_.offset = this->data_.offset;
    t->data_.memory = std::make_shared<Memory>(this->data_.memory->data(), this->data_.memory->size(), this->data_.memory->device(), nullptr);
    t->to_blob_mark_ = true;
    return Tensor{t};
}

Tensor TensorImpl::resume_from_blob_() const {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape(), strides(), dtype()));
    t->data_.offset = this->data_.offset;
    if (to_blob_mark_) {
        t->data_.memory = context::reinstantiateBlob(this->data_.memory);
    } else {
        t->data_.memory = this->data_.memory;
    }

    return Tensor{t};
}

} // namespace infinicore
