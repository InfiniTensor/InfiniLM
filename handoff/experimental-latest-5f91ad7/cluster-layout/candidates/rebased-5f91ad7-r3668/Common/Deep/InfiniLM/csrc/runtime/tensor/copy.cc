#include "execution/submit.hpp"
#include "runtime/context.hpp"
#include "runtime/dtype.hpp"
#include "runtime/owner_drain_fail_stop.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"

#include <algorithm>
#include <atomic>
#include "cpu_strided_copy_r2515.hpp"
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::runtime {
namespace {
std::atomic<std::uint64_t> copy_calls_r2515{0}, copy_blocks_r2515{0}, copy_bulk_bytes_r2515{0}, copy_leaves_r2515{0}, copy_scalar_bytes_r2515{0};

void copyCpuStrided(std::byte *dst,
                    const std::byte *src,
                    const Shape &shape,
                    const Strides &dst_strides,
                    const Strides &src_strides,
                    size_t element_size,
                    size_t dim = 0) {
    detail::CpuCopyStatsR2515 stats;
    detail::copy_cpu_strided_r2515(dst, src, shape, dst_strides, src_strides, element_size, dim, stats);
    copy_calls_r2515.fetch_add(1, std::memory_order_relaxed);
    copy_blocks_r2515.fetch_add(stats.blocks, std::memory_order_relaxed);
    copy_bulk_bytes_r2515.fetch_add(stats.block_bytes, std::memory_order_relaxed);
    copy_leaves_r2515.fetch_add(stats.scalar_leaves, std::memory_order_relaxed);
    copy_scalar_bytes_r2515.fetch_add(stats.scalar_bytes, std::memory_order_relaxed);
}

} // namespace

Tensor TensorImpl::to(Device device) const {
    if (device == data_.memory->device()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    }
    auto source = allocation_owner();
    if (!source || source->device() != this->device()) {
        source = context::resolveRuntime(this->device());
    }
    auto destination = context::resolveRuntime(device);
    return to(*source, *destination);
}

Tensor TensorImpl::to(Runtime &source_runtime, Runtime &destination_runtime) const {
    source_runtime.assertOwnerThread();
    destination_runtime.assertOwnerThread();
    if (device().type() != Device::Type::kCpu && source_runtime.device() != device()) {
        throw std::invalid_argument("to Runtime and source tensor devices differ");
    }
    if (&source_runtime == &destination_runtime && device() == destination_runtime.device()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    }
    if (device().type() != Device::Type::kCpu && destination_runtime.device().type() == Device::Type::kCpu) {
        return to(source_runtime, destination_runtime.device());
    }
    auto result = Tensor::empty(destination_runtime, shape(), dtype());
    result->copy_from(destination_runtime, source_runtime,
                      Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
    return result;
}

Tensor TensorImpl::to(Runtime &runtime, Device target) const {
    runtime.assertOwnerThread();
    if (runtime.device() != device()) {
        throw std::invalid_argument("to Runtime and source tensor devices differ");
    }
    if (target == device()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    }
    if (target.type() != Device::Type::kCpu || device().type() == Device::Type::kCpu) {
        throw std::invalid_argument("explicit to only supports source-owned GPU to CPU transfer");
    }
    if (runtime.graphCaptureState() != ::infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("D2H host observation is not allowed during graph recording");
    }
    // Host-only owning storage: do not bind a second CPU Runtime.
    auto storage = std::make_shared<std::vector<std::byte>>(nbytes());
    auto result = Tensor::from_blob(storage->data(), shape(), dtype(), target, storage);
    result->copy_from(runtime, Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
    return result;
}

void TensorImpl::copy_from(Tensor src) {
    if (!src) {
        throw std::invalid_argument("copy_from requires a source tensor");
    }
    auto source = src->allocation_owner();
    auto destination = allocation_owner();
    const bool external_gpu = !source && src->device().type() != Device::Type::kCpu
                           && device().type() != Device::Type::kCpu;
    // CPU execution is independent of a pinned allocation's accelerator owner.
    // Ownerless external data is already completed by the frontend/caller.
    if (!source || source->device() != src->device()) {
        source = context::resolveRuntime(src->device());
    }
    if (!destination || destination->device() != device()) {
        destination = context::resolveRuntime(device());
    }
    if (external_gpu) {
        if (destination->graphCaptureState() != ::infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
            throw std::logic_error("completed external GPU copies are not allowed during graph recording");
        }
        try {
            copy_from(*destination, *source, src);
            destination->syncStream();
        } catch (...) {
            try {
                destination->syncStream();
            } catch (...) { detail::require_proven_owner_drain(true, false); }
            throw;
        }
    } else {
        copy_from(*destination, *source, std::move(src));
    }
}

void TensorImpl::copy_from(Runtime &destination, Runtime &source, Tensor src) {
    destination.assertOwnerThread();
    source.assertOwnerThread();
    if (!src) {
        throw std::invalid_argument("copy_from requires a source tensor");
    }
    if ((device().type() != Device::Type::kCpu && destination.device() != device())
        || (src->device().type() != Device::Type::kCpu && source.device() != src->device())) {
        throw std::invalid_argument("copy_from explicit owners and tensor devices differ");
    }
    if (src->shape() != shape() || src->dtype() != dtype()) {
        throw std::runtime_error("Cannot copy from tensor with different shape or dtype");
    }
    if (src->device().type() == Device::Type::kCpu) {
        copy_from(destination, std::move(src));
    } else if (device().type() == Device::Type::kCpu) {
        copy_from(source, std::move(src));
    } else if (&source == &destination) {
        copy_from(destination, std::move(src));
    } else {
        if (source.graphCaptureState() != ::infinilm::execution::graph::GraphManager::CaptureState::kInactive
            || destination.graphCaptureState() != ::infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
            throw std::logic_error("cross-owner copies are not allowed during graph recording");
        }
        if (source.device() == destination.device()) {
            source.syncStream();
            try {
                copy_from(destination, src);
                destination.syncStream(); // Do not return src to another pool before its read completes.
            } catch (...) {
                try {
                    destination.syncStream();
                } catch (...) { detail::require_proven_owner_drain(true, false); }
                throw;
            }
        } else {
            // The existing completed D2H/H2D leaves own gather/scatter and unwind.
            auto staging = src->to(source, Device{Device::Type::kCpu});
            copy_from(destination, std::move(staging));
        }
    }
}

Tensor TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else {
        auto runtime = allocation_owner();
        const bool external_gpu = !runtime && device().type() != Device::Type::kCpu;
        if (!runtime || runtime->device() != device()) {
            runtime = context::resolveRuntime(device());
        }
        if (external_gpu) {
            if (runtime->graphCaptureState() != ::infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
                throw std::logic_error("completed external GPU materialization is not allowed during graph recording");
            }
            auto source = Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
            try {
                auto result = contiguous(*runtime);
                runtime->syncStream();
                return result;
            } catch (...) {
                try {
                    runtime->syncStream();
                } catch (...) { detail::require_proven_owner_drain(true, false); }
                throw;
            }
        }
        return contiguous(*runtime);
    }
}

Tensor TensorImpl::contiguous(Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (runtime.device() != device()) {
        throw std::invalid_argument("contiguous runtime and tensor devices differ");
    }
    auto source = Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    if (is_contiguous()) {
        return source;
    }
    auto result = Tensor::empty(runtime, shape(), dtype());
    if (device().type() == Device::Type::kCpu) {
        // Newly allocated output cannot overlap the input: no staging needed.
        copyCpuStrided(result->data(), data(), shape(), result->strides(),
                       strides(), element_size());
    } else {
        infinilm::execution::submit<infini::ops::Copy>(
            runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
            source, false, result);
    }
    return result;
}

void TensorImpl::copy_from(Runtime &runtime, Tensor src) {
    runtime.assertOwnerThread();
    if (!src) {
        throw std::invalid_argument("copy_from requires a source tensor");
    }
    const bool d2h = this->device().type() == Device::Type::kCpu
                  && src->device().type() != Device::Type::kCpu;
    if ((d2h && runtime.device() != src->device())
        || (!d2h && (runtime.device() != this->device() || (src->device() != this->device() && src->device().type() != Device::Type::kCpu)))) {
        throw std::invalid_argument("copy_from Runtime and tensor devices differ");
    }
    if (src->shape() != this->shape()) {
        throw std::runtime_error(
            "Cannot copy from tensor with different shape. Src: " + src->info() + " Dst: " + this->info());
    }
    if (src->dtype() != this->dtype()) {
        throw std::runtime_error(
            "Cannot copy from tensor with different dtype. Src: " + src->info() + " Dst: " + this->info());
    }

    if (d2h) {
        if (runtime.graphCaptureState() != ::infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
            throw std::logic_error("D2H host observation is not allowed during graph recording");
        }
        // These resources must survive the catch-side completion proof.
        auto contiguous_src = src;
        auto destination = Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
        std::vector<std::byte> gathered;
        try {
            if (!src->is_contiguous()) {
                contiguous_src = Tensor::empty(runtime, src->shape(), src->dtype());
                infinilm::execution::submit<infini::ops::Copy>(
                    runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                    src, false, contiguous_src);
            }
            if (this->is_contiguous()) {
                runtime.memcpyD2H(this->data(), contiguous_src->data(), this->nbytes());
            } else {
                gathered.resize(this->nbytes());
                runtime.memcpyD2H(gathered.data(), contiguous_src->data(), gathered.size());
                auto packed = Tensor::from_blob(gathered.data(), shape(), dtype(), device());
                copyCpuStrided(data(), packed->data(), shape(), strides(),
                               packed->strides(), element_size());
            }
        } catch (...) {
            try {
                runtime.syncStream();
            } catch (...) {
                detail::require_proven_owner_drain(true, false);
            }
            throw;
        }
        return;
    }

    auto dst = Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    if (this->device() == src->device()) {
        if (this->device().type() != Device::Type::kCpu) {
            infinilm::execution::submit<infini::ops::Copy>(
                runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                src, false, dst);
            return;
        }
        if (this->is_contiguous() && src->is_contiguous()) {
            if (this->nbytes() != 0) {
                std::memmove(this->data(), src->data(), this->nbytes());
            }
            return;
        }
        // Gather before scattering so overlapping CPU views retain the old
        // copy_from behavior without selecting a thread-local Runtime.
        std::vector<std::byte> gathered(this->nbytes());
        auto contiguous = Tensor::from_blob(
            gathered.data(), this->shape(), this->dtype(), this->device());
        copyCpuStrided(contiguous->data(), src->data(), this->shape(),
                       contiguous->strides(), src->strides(), this->element_size());
        copyCpuStrided(this->data(), contiguous->data(), this->shape(),
                       this->strides(), contiguous->strides(), this->element_size());
        return;
    }

    // Only CPU -> Runtime device crosses devices in this explicit contract.
    // Synchronize H2D before releasing a caller-owned or gathered host buffer.
    std::vector<std::byte> gathered;
    Tensor local_src;
    try {
        if (this->is_contiguous()
            && this->device().type() == Device::Type::kNvidia
            && runtime.graphCaptureState() == ::infinilm::execution::graph::GraphManager::CaptureState::kInactive
            && (src->dtype() == DataType::kFloat32
                || src->dtype() == DataType::kFloat16
                || src->dtype() == DataType::kBFloat16)
            && src->ndim() == 2 && src->shape()[0] > 1 && src->shape()[1] > 1
            && src->strides()[0] == 1 && src->strides()[1] > 0
            && static_cast<size_t>(src->strides()[1]) >= src->shape()[0]) {
            const auto span = storage_span_bytes(src->shape(), src->strides(), src->element_size());
            if (src->data_.offset > src->data_.memory->size()
                || span > src->data_.memory->size() - src->data_.offset) {
                throw std::out_of_range("strided H2D source exceeds underlying storage");
            }
            // ceil(span / 3) avoids overflowing the three-times-logical bound.
            if (span / 3 + (span % 3 != 0) <= src->nbytes()) {
                local_src = Tensor::strided_empty(runtime, src->shape(), src->strides(), src->dtype());
                runtime.memcpyH2D(local_src->data(), src->data(), span, false);
                infinilm::execution::submit<infini::ops::Copy>(
                    runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                    local_src, false, dst);
                runtime.syncStream(); // Keep src/dst/local_src alive through completion.
                return;
            }
        }
        const std::byte *host_data = src->data();
        if (!src->is_contiguous()) {
            gathered.resize(this->nbytes());
            auto contiguous = Tensor::from_blob(
                gathered.data(), src->shape(), src->dtype(), src->device());
            copyCpuStrided(contiguous->data(), src->data(), src->shape(),
                           contiguous->strides(), src->strides(), src->element_size());
            host_data = gathered.data();
        }
        if (this->is_contiguous()) {
            runtime.memcpyH2D(this->data(), host_data, this->nbytes(), false);
        } else {
            local_src = Tensor::empty(runtime, this->shape(), this->dtype());
            runtime.memcpyH2D(local_src->data(), host_data, this->nbytes(), false);
            infinilm::execution::submit<infini::ops::Copy>(
                runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                local_src, false, dst);
        }
    } catch (...) {
        // Keep host gather and device temporary alive until completion is known.
        try {
            runtime.syncStream();
        } catch (...) {
            detail::require_proven_owner_drain(true, false);
        }
        throw;
    }
}

} // namespace infinilm::runtime

extern "C" int modern_cpu_copy_stats_r2515(std::uint64_t*out,std::size_t n){
 if(!out||n!=5)return 1;
 using namespace infinilm::runtime;
 out[0]=copy_calls_r2515.load(std::memory_order_relaxed);
 out[1]=copy_blocks_r2515.load(std::memory_order_relaxed);
 out[2]=copy_bulk_bytes_r2515.load(std::memory_order_relaxed);
 out[3]=copy_leaves_r2515.load(std::memory_order_relaxed);
 out[4]=copy_scalar_bytes_r2515.load(std::memory_order_relaxed);return 0;
}
