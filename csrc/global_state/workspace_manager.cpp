#include "workspace_manager.hpp"

#include "../utils.hpp"
#include "parallel_state.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace infinilm::global_state {

namespace {

constexpr size_t k_scratch_align_bytes = 512;

size_t compute_numel(const infinicore::Shape &shape) {
    size_t numel = 1;
    for (const auto dim : shape) {
        numel *= dim;
    }
    return numel;
}

size_t align_up(size_t n, size_t alignment = k_scratch_align_bytes) {
    return (n + alignment - 1) & ~(alignment - 1);
}

size_t compute_aligned_bytes(const infinicore::Shape &shape, const infinicore::DataType &dtype) {
    return align_up(compute_numel(shape) * infinicore::dsize(dtype));
}

} // namespace

void WorkspaceManager::register_buffer(const std::string &name,
                                       const infinicore::Shape &shape,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    _register_buffer_impl(name, total_bytes_, shape, dtype, device, true);
}

void WorkspaceManager::register_buffer(const std::string &name,
                                       size_t offset,
                                       const infinicore::Shape &shape,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    ASSERT(0 == offset);
    _register_buffer_impl(name, offset, shape, dtype, device, false);
}

infinicore::Tensor WorkspaceManager::_make_runtime_view(size_t offset,
                                                        const infinicore::Shape &shape,
                                                        const infinicore::DataType &dtype,
                                                        const infinicore::Device &device) {
    auto *base_ptr = scratch_buffer_->data() + offset;
    return infinicore::Tensor::from_blob(static_cast<void *>(base_ptr), shape, dtype, device);
}

infinicore::Tensor WorkspaceManager::get_buffer(const std::string &buffer_name,
                                                const infinicore::Shape &shape,
                                                const infinicore::DataType &dtype,
                                                const infinicore::Device &device) {
    ASSERT(finalized_);
    ASSERT(!scratch_buffer_.empty());

    auto cached = runtime_buffers_.find(buffer_name);
    if (cached != runtime_buffers_.end()) {
        return cached->second;
    }

    auto &rank_device = get_tensor_model_parallel_rank_info().device;
    const size_t aligned_bytes = compute_aligned_bytes(shape, dtype);

    auto registered = registrations_.find(buffer_name);
    if (registered != registrations_.end()) {
        const auto &reg = registered->second;
        auto tensor = _make_runtime_view(reg.offset, shape, dtype, rank_device);
        runtime_buffers_.emplace(buffer_name, tensor);
        return tensor;
    }

    const size_t offset = scratch_buffer_offset_;
    ASSERT(offset + aligned_bytes <= total_bytes_);

    auto tensor = _make_runtime_view(offset, shape, dtype, rank_device);
    runtime_buffers_.emplace(buffer_name, tensor);
    scratch_buffer_offset_ += aligned_bytes;
    return tensor;
}

infinicore::Tensor WorkspaceManager::get_buffer(const std::string &buffer_name,
                                                size_t offset,
                                                const infinicore::Shape &shape,
                                                const infinicore::DataType &dtype,
                                                const infinicore::Device &device) {
    ASSERT(finalized_);
    ASSERT(!scratch_buffer_.empty());

    auto cached = runtime_buffers_.find(buffer_name);
    if (cached != runtime_buffers_.end()) {
        return cached->second;
    }

    auto &rank_device = get_tensor_model_parallel_rank_info().device;
    const size_t aligned_bytes = compute_aligned_bytes(shape, dtype);

    auto registered = registrations_.find(buffer_name);
    if (registered != registrations_.end()) {
        const auto &reg = registered->second;
        auto tensor = _make_runtime_view(reg.offset, shape, dtype, rank_device);
        runtime_buffers_.emplace(buffer_name, tensor);
        return tensor;
    }

    ASSERT(offset + aligned_bytes <= total_bytes_);

    auto tensor = _make_runtime_view(offset, shape, dtype, rank_device);
    runtime_buffers_.emplace(buffer_name, tensor);
    return tensor;
}

void WorkspaceManager::reset_runtime_buffers() {
    ASSERT(finalized_);
    scratch_buffer_offset_ = 0;
    runtime_buffers_.clear();
}

void WorkspaceManager::finalize_and_bind() {
    ASSERT(!finalized_);
    runtime_buffers_.clear();
    scratch_buffer_offset_ = 0;

    if (total_bytes_ == 0) {
        finalized_ = true;
        return;
    }

    auto &rank_device = get_tensor_model_parallel_rank_info().device;

    scratch_buffer_ = infinicore::Tensor::empty({total_bytes_}, infinicore::DataType::U8, rank_device);

    spdlog::info("WorkspaceManager: finalize_and_bind {:.3f} MB", total_bytes_ / 1024.0 / 1024.0);

    for (auto &entry : registrations_) {
        auto &reg = entry.second;
        auto *base_ptr = scratch_buffer_->data() + reg.offset;
        ASSERT(rank_device == reg.device);
        reg.bound_view = infinicore::Tensor::from_blob(static_cast<void *>(base_ptr), reg.shape, reg.dtype, rank_device);
    }

    scratch_buffer_offset_ = 0;
    finalized_ = true;
}

void WorkspaceManager::log_registrations() const {
    std::vector<std::string> names;
    names.reserve(registrations_.size());
    for (const auto &entry : registrations_) {
        names.push_back(entry.first);
    }
    std::sort(names.begin(), names.end(), [this](const std::string &a, const std::string &b) {
        return registrations_.at(a).offset < registrations_.at(b).offset;
    });

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "\n========== WorkspaceManager registrations ==========\n";
    oss << "  " << std::setw(16) << std::left << "finalized:" << finalized_ << "\n";
    oss << "  " << std::setw(16) << std::left << "slots:" << registrations_.size() << "\n";
    oss << "  " << std::setw(16) << std::left << "runtime_buffers:" << runtime_buffers_.size() << "\n";
    oss << "  " << std::setw(16) << std::left << "scratch_bytes:"
        << total_bytes_ << " (" << (total_bytes_ / 1024.0 / 1024.0) << " MB)\n";
    oss << "  " << std::setw(16) << std::left << "scratch_buffer_offset_:"
        << scratch_buffer_offset_ << " (" << (scratch_buffer_offset_ / 1024.0 / 1024.0) << " MB)\n";
    oss << "  note: scratch_bytes=max span; registered slots may overlap.\n";
    oss << "----------------------------------------------------\n";

    auto memory_end = [](const BufferRegistration &reg) {
        return reg.offset + reg.aligned_bytes;
    };
    auto ranges_overlap = [](size_t a_start, size_t a_end, size_t b_start, size_t b_end) {
        return a_start < b_end && b_start < a_end;
    };

    for (size_t slot_idx = 0; slot_idx < names.size(); ++slot_idx) {
        const auto &name = names[slot_idx];
        const auto &reg = registrations_.at(name);
        const size_t mem_start = reg.offset;
        const size_t mem_end = memory_end(reg);

        std::string shape_str = "[";
        for (size_t i = 0; i < reg.shape.size(); ++i) {
            if (i > 0) {
                shape_str += ", ";
            }
            shape_str += std::to_string(reg.shape[i]);
        }
        shape_str += "]";

        std::string overlap_str = "none";
        {
            std::ostringstream overlap_oss;
            bool first = true;
            for (size_t other_idx = 0; other_idx < names.size(); ++other_idx) {
                if (other_idx == slot_idx) {
                    continue;
                }
                const auto &other = registrations_.at(names[other_idx]);
                if (ranges_overlap(mem_start, mem_end, other.offset, memory_end(other))) {
                    if (!first) {
                        overlap_oss << ", ";
                    }
                    overlap_oss << "slot " << other_idx;
                    first = false;
                }
            }
            if (!first) {
                overlap_str = overlap_oss.str();
            }
        }

        oss << "  [slot " << slot_idx << "]\n";
        oss << "    " << std::setw(16) << std::left << "layout:"
            << (reg.is_bump_tail ? "bump" : "pinned@0") << "\n";
        oss << "    " << std::setw(16) << std::left << "memory:"
            << "[" << mem_start << ", " << mem_end << ") "
            << "(" << (reg.aligned_bytes / 1024.0 / 1024.0) << " MB)\n";
        oss << "    " << std::setw(16) << std::left << "overlaps:" << overlap_str << "\n";
        oss << "    " << std::setw(16) << std::left << "name:" << name << "\n";
        oss << "    " << std::setw(16) << std::left << "shape:" << shape_str << "\n";
        oss << "    " << std::setw(16) << std::left << "dtype:" << infinicore::toString(reg.dtype) << "\n";
        oss << "    " << std::setw(16) << std::left << "device:" << reg.device.toString() << "\n";
        oss << "    " << std::setw(16) << std::left << "bound:" << finalized_ << "\n";
        if (slot_idx + 1 < names.size()) {
            oss << "\n";
        }
    }
    oss << "====================================================\n";

    spdlog::info("{}", oss.str());
}

void WorkspaceManager::_register_buffer_impl(const std::string &name,
                                             size_t offset,
                                             const infinicore::Shape &shape,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device,
                                             bool bump_tail) {
    ASSERT(!finalized_);
    ASSERT(device == get_tensor_model_parallel_rank_info().device);

    const size_t aligned_bytes = compute_aligned_bytes(shape, dtype);

    if (registrations_.find(name) == registrations_.end()) {
        BufferRegistration reg;
        reg.offset = offset;
        reg.aligned_bytes = aligned_bytes;
        reg.is_bump_tail = bump_tail;
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
    ASSERT(reg.is_bump_tail == bump_tail);
    ASSERT(reg.aligned_bytes == aligned_bytes);
    ASSERT(reg.shape == shape);
    ASSERT(reg.dtype == dtype);
    ASSERT(reg.device == device);
}

} // namespace infinilm::global_state
