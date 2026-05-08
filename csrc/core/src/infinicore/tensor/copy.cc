#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"
#include "../../utils/rearrange.h"

#include <algorithm>
#include <cstring>
#include <iostream>
namespace infinicore {
namespace {
void rearrange_cpu(Tensor dst, Tensor src) {
    utils::rearrange(
        dst->data(),
        src->data(),
        dst->shape().data(),
        dst->strides().data(),
        src->strides().data(),
        dst->ndim(),
        dst->element_size());
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
    if (this->device() == src->device()) {
        auto dst = Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
        if (this->device().getType() == Device::Type::CPU) {
            rearrange_cpu(dst, src);
        } else if (this->is_contiguous() && src->is_contiguous()) {
            context::setDevice(this->device());
            context::memcpyD2D(this->data(), src->data(), std::min(this->nbytes(), src->nbytes()));
        } else {
            throw std::runtime_error("Device-side non-contiguous copy requires an external InfiniOps rearrange wrapper");
        }
    } else {
        if (!src->is_contiguous()) {
            src = src->contiguous();
        }

        // Use nbytes() to get the actual tensor size, not the full memory size
        size_t copy_size = std::min(this->nbytes(), src->nbytes());
        if (this->device().getType() == Device::Type::CPU) {
            if (this->is_contiguous()) {
                context::setDevice(src->device());
                context::memcpyD2H(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::setDevice(src->device());
                context::memcpyD2H(local_src->data(), src->data(), copy_size);
                rearrange_cpu(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        } else if (src->device().getType() == Device::Type::CPU) {
            context::setDevice(this->device());
            if (this->is_contiguous()) {
                context::memcpyH2D(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyH2D(local_src->data(), src->data(), copy_size);
                throw std::runtime_error("Device-side non-contiguous H2D copy requires an external InfiniOps rearrange wrapper");
            }
        }
    }
}

Tensor TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    }
    if (this->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("Device-side contiguous() for non-contiguous tensors requires an external InfiniOps rearrange wrapper");
    }
    auto out = Tensor::empty(this->shape(), this->dtype(), this->device());
    rearrange_cpu(out, Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
    return out;
}

} // namespace infinicore
