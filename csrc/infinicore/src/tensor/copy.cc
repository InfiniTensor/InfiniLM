#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

#include <algorithm>
#include <cstring>

namespace infinicore {
namespace {

void copyCpuStrided(std::byte *dst,
                    const std::byte *src,
                    const Shape &shape,
                    const Strides &dst_strides,
                    const Strides &src_strides,
                    size_t element_size,
                    size_t dim = 0) {
    if (dim == shape.size()) {
        std::memmove(dst, src, element_size);
        return;
    }

    const auto byte_size = static_cast<std::ptrdiff_t>(element_size);
    for (size_t index = 0; index < shape[dim]; ++index) {
        copyCpuStrided(
            dst + static_cast<std::ptrdiff_t>(index) * dst_strides[dim] * byte_size,
            src + static_cast<std::ptrdiff_t>(index) * src_strides[dim] * byte_size,
            shape,
            dst_strides,
            src_strides,
            element_size,
            dim + 1);
    }
}

} // namespace

Tensor TensorImpl::to(Device device) const {
    if (device == data_.memory->device()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else {
        std::shared_ptr<TensorImpl> _t = empty(meta_.shape, meta_.dtype, device);
        _t->copy_from(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
        return Tensor(_t);
    }
}

void TensorImpl::copy_from(Tensor src) {
    if (src->shape() != this->shape()) {
        throw std::runtime_error(
            "Cannot copy from tensor with different shape. Src: " + src->info() + " Dst: " + this->info());
    }
    if (src->dtype() != this->dtype()) {
        throw std::runtime_error(
            "Cannot copy from tensor with different dtype. Src: " + src->info() + " Dst: " + this->info());
    }
    if (this->device() == src->device()) {
        if (this->device().type() == Device::Type::kCpu) {
            if (this->is_contiguous() && src->is_contiguous()) {
                if (this->nbytes() != 0) {
                    std::memmove(this->data(), src->data(), this->nbytes());
                }
            } else {
                auto host_staging = Tensor::empty(
                    this->shape(), this->dtype(), Device{Device::Type::kCpu});
                copyCpuStrided(
                    host_staging->data(), src->data(), this->shape(),
                    host_staging->strides(), src->strides(), this->element_size());
                copyCpuStrided(
                    this->data(), host_staging->data(), this->shape(),
                    this->strides(), host_staging->strides(), this->element_size());
            }
        } else {
            op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), src);
        }
    } else {
        if (!src->is_contiguous()) {
            src = src->contiguous();
        }

        // Use nbytes() to get the actual tensor size, not the full memory size
        size_t copy_size = std::min(this->nbytes(), src->nbytes());
        if (this->device().type() == Device::Type::kCpu) {
            if (this->is_contiguous()) {
                context::setDevice(src->device());
                context::memcpyD2H(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::setDevice(src->device());
                context::memcpyD2H(local_src->data(), src->data(), copy_size);
                this->copy_from(local_src);
            }
        } else if (src->device().type() == Device::Type::kCpu) {
            context::setDevice(this->device());
            // copy_from does not retain the host source after it returns.
            if (this->is_contiguous()) {
                context::memcpyH2D(this->data(), src->data(), copy_size, false);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyH2D(local_src->data(), src->data(), copy_size, false);
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        } else {
            auto host_staging = Tensor::empty(
                this->shape(), this->dtype(), Device{Device::Type::kCpu});
            host_staging->copy_from(src);
            this->copy_from(host_staging);
        }
    }
}

Tensor TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else if (device().type() == Device::Type::kCpu) {
        auto result = Tensor::empty(shape(), dtype(), device());
        result->copy_from(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
        return result;
    } else {
        return op::rearrange(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
    }
}

} // namespace infinicore
