#include "workspace_manager.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::global_state {
namespace {

constexpr size_t WORKSPACE_ALIGNMENT = 512;

size_t align_up(size_t value) {
    if (value > std::numeric_limits<size_t>::max() - (WORKSPACE_ALIGNMENT - 1)) {
        throw std::overflow_error("WorkspaceManager size overflow");
    }
    return (value + WORKSPACE_ALIGNMENT - 1) & ~(WORKSPACE_ALIGNMENT - 1);
}

} // namespace

size_t WorkspaceManager::aligned_nbytes(const infinicore::Shape &shape,
                                        const infinicore::DataType &dtype) {
    size_t numel = 1;
    for (size_t dim : shape) {
        if (dim != 0 && numel > std::numeric_limits<size_t>::max() / dim) {
            throw std::overflow_error("WorkspaceManager shape overflow");
        }
        numel *= dim;
    }
    const size_t element_size = infinicore::dsize(dtype);
    if (element_size != 0 && numel > std::numeric_limits<size_t>::max() / element_size) {
        throw std::overflow_error("WorkspaceManager byte-size overflow");
    }
    return align_up(numel * element_size);
}

void WorkspaceManager::reserve_slot(const std::string &name,
                                    const infinicore::Shape &max_shape,
                                    const infinicore::DataType &dtype,
                                    const infinicore::Device &device) {
    if (finalized_) {
        throw std::logic_error("WorkspaceManager cannot reserve slots after finalize");
    }
    const size_t capacity = aligned_nbytes(max_shape, dtype);
    auto [it, inserted] = slots_.try_emplace(name, Slot{0, capacity, dtype, device});
    if (!inserted) {
        if (it->second.dtype != dtype || it->second.device != device) {
            throw std::invalid_argument("WorkspaceManager slot metadata mismatch: " + name);
        }
        it->second.capacity_bytes = std::max(it->second.capacity_bytes, capacity);
    }
}

void WorkspaceManager::finalize() {
    if (finalized_) {
        throw std::logic_error("WorkspaceManager is already finalized");
    }
    total_bytes_ = 0;
    std::optional<infinicore::Device> device;
    for (auto &[name, slot] : slots_) {
        (void)name;
        if (device.has_value() && device.value() != slot.device) {
            throw std::invalid_argument("WorkspaceManager slots must use one device");
        }
        device = slot.device;
        slot.offset = total_bytes_;
        if (slot.capacity_bytes > std::numeric_limits<size_t>::max() - total_bytes_) {
            throw std::overflow_error("WorkspaceManager total size overflow");
        }
        total_bytes_ += slot.capacity_bytes;
    }
    if (total_bytes_ > 0) {
        storage_ = infinicore::Tensor::empty(
            {total_bytes_}, infinicore::DataType::U8, device.value());
    }
    finalized_ = true;
    spdlog::info("WorkspaceManager allocated {:.3f} MiB across {} slots",
                 total_bytes_ / 1024.0 / 1024.0, slots_.size());
}

void WorkspaceManager::reset_runtime_buffers() {
    if (!finalized_) {
        throw std::logic_error("WorkspaceManager must be finalized before reset");
    }
    runtime_views_.clear();
}

infinicore::Tensor WorkspaceManager::get_buffer(
    const std::string &name,
    const infinicore::Shape &shape,
    const infinicore::DataType &dtype,
    const infinicore::Device &device) {
    if (!finalized_) {
        throw std::logic_error("WorkspaceManager must be finalized before use");
    }
    const auto slot_it = slots_.find(name);
    if (slot_it == slots_.end()) {
        throw std::out_of_range("WorkspaceManager slot is not registered: " + name);
    }
    const Slot &slot = slot_it->second;
    if (slot.dtype != dtype || slot.device != device) {
        throw std::invalid_argument("WorkspaceManager slot metadata mismatch: " + name);
    }
    const size_t requested_bytes = aligned_nbytes(shape, dtype);
    if (requested_bytes > slot.capacity_bytes) {
        throw std::out_of_range(
            "WorkspaceManager slot capacity exceeded: " + name
            + ", requested=" + std::to_string(requested_bytes)
            + ", capacity=" + std::to_string(slot.capacity_bytes));
    }
    auto *ptr = storage_->data() + slot.offset;
    auto tensor = infinicore::Tensor::from_blob(
        static_cast<void *>(ptr), shape, dtype, device);
    runtime_views_[name] = tensor;
    return tensor;
}

} // namespace infinilm::global_state
