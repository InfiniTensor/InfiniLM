#pragma once

#include "../models/infinilm_model.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace infinilm::global_state {

/**
 * @brief Unified GPU inference workspace manager.
 *
 * Phase 1: modules register buffer layouts via ``register_buffer``.
 * Phase 2/3: ``finalize_and_bind`` allocates ``scratch_buffer_`` and binds views.
 */
class WorkspaceManager {
public:
    using BindFn = std::function<void(const infinicore::Tensor &)>;

    WorkspaceManager() = default;
    ~WorkspaceManager() = default;

    /**
     * @brief Register a buffer appended at the current scratch_buffer tail.
     *
     * @param name Unique cache key; duplicate keys share one slot.
     * @param shape Tensor shape for the bound view.
     * @param dtype Element type of the bound view.
     * @param device Device on which scratch_buffer is allocated.
     * @param bind_fn Callback invoked in ``finalize_and_bind`` with the bound view.
     */
    void register_buffer(const std::string &name,
                         const infinicore::Shape &shape,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device,
                         BindFn bind_fn) {
        register_buffer_impl(name, total_bytes_, shape, dtype, device, std::move(bind_fn), true);
    }

    /**
     * @brief Register a buffer pinned at a fixed byte offset.
     *
     * @param name Unique cache key; duplicate keys share one slot.
     * @param offset Byte offset in scratch_buffer (currently only 0 is supported).
     * @param shape Tensor shape for the bound view.
     * @param dtype Element type of the bound view.
     * @param device Device on which scratch_buffer is allocated.
     * @param bind_fn Callback invoked in ``finalize_and_bind`` with the bound view.
     */
    void register_buffer(const std::string &name,
                         size_t offset,
                         const infinicore::Shape &shape,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device,
                         BindFn bind_fn) {
        ASSERT(0 == offset);
        register_buffer_impl(name, offset, shape, dtype, device, std::move(bind_fn), false);
    }

    /**
     * @brief Allocate scratch_buffer and run all registered bind callbacks.
     *
     * @param device Device on which scratch_buffer is allocated.
     */
    void finalize_and_bind(const infinicore::Device &device) {
        ASSERT(!finalized_);
        if (total_bytes_ == 0) {
            finalized_ = true;
            return;
        }

        ASSERT(device.getType() != infinicore::Device::Type::CPU);

        scratch_buffer_ = infinicore::Tensor::empty({total_bytes_}, infinicore::DataType::U8, device);

        spdlog::info("WorkspaceManager: finalize_and_bind {:.3f} MB", total_bytes_ / 1024.0 / 1024.0);

        for (auto &[name, reg] : registrations_) {
            auto *base_ptr = scratch_buffer_->data() + reg.offset;
            auto view = infinicore::Tensor::from_blob(static_cast<void *>(base_ptr), reg.shape, reg.dtype, device);
            inference_buffers_[name] = view;
            for (auto &bind_fn : reg.bind_callbacks) {
                bind_fn(view);
            }
        }

        finalized_ = true;
    }

private:
    /** @brief Metadata for one registered region in scratch_buffer. */
    struct BufferRegistration {
        size_t offset{0};
        size_t aligned_bytes{0};
        infinicore::Shape shape;
        infinicore::DataType dtype;
        infinicore::Device device;
        std::vector<BindFn> bind_callbacks;
    };

    void register_buffer_impl(const std::string &name,
                              size_t offset,
                              const infinicore::Shape &shape,
                              const infinicore::DataType &dtype,
                              const infinicore::Device &device,
                              BindFn bind_fn,
                              bool bump_tail) {
        ASSERT(!finalized_);
        ASSERT(device.getType() != infinicore::Device::Type::CPU);

        auto compute_numel = [](const infinicore::Shape &shape) {
            size_t numel = 1;
            for (const auto dim : shape) {
                numel *= dim;
            }
            return numel;
        };

        auto align_up = [](size_t n, size_t alignment = 512) {
            return (n + alignment - 1) & ~(alignment - 1);
        };

        const size_t actual_bytes = compute_numel(shape) * infinicore::dsize(dtype);
        const size_t aligned_bytes = align_up(actual_bytes);

        if (registrations_.find(name) == registrations_.end()) {
            BufferRegistration reg;
            reg.offset = offset;
            reg.aligned_bytes = aligned_bytes;
            reg.shape = shape;
            reg.dtype = dtype;
            reg.device = device;

            if (bump_tail) {
                total_bytes_ += aligned_bytes;
            } else {
                total_bytes_ = std::max(total_bytes_, offset + aligned_bytes);
            }
            registrations_.emplace(name, std::move(reg));
        }

        auto &reg = registrations_.at(name);
        ASSERT(reg.aligned_bytes == aligned_bytes);
        ASSERT(reg.shape == shape);
        ASSERT(reg.dtype == dtype);
        ASSERT(reg.device == device);
        reg.bind_callbacks.push_back(std::move(bind_fn));
    }

    size_t total_bytes_{0};
    bool finalized_{false};
    infinicore::Tensor scratch_buffer_;
    std::unordered_map<std::string, BufferRegistration> registrations_;
    std::unordered_map<std::string, infinicore::Tensor> inference_buffers_;
};

}; // namespace infinilm::global_state
