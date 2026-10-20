#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace infinilm::parallel {

// Existing topology: contiguous TP ranks within each PP stage. No model,
// request, runtime or communicator ownership belongs in this value object.
class Topology {
public:
    Topology(size_t local_tp_size, int pp_size = 1, int pp_stage = 0)
        : pp_size_(pp_size), pp_stage_(pp_stage) {
        if (pp_size < 1) {
            throw std::invalid_argument("pipeline parallel size must be at least 1");
        }
        if (pp_stage < 0 || pp_stage >= pp_size) {
            throw std::invalid_argument("pipeline parallel stage must be in [0, pp_size)");
        }
        const auto limit = static_cast<size_t>(std::numeric_limits<int>::max());
        if (local_tp_size == 0) {
            throw std::invalid_argument("tensor parallel device list must not be empty");
        }
        if (local_tp_size > limit) {
            throw std::invalid_argument("tensor parallel world size exceeds the InfiniCCL rank limit");
        }
        if (local_tp_size > limit / static_cast<size_t>(pp_size)) {
            throw std::invalid_argument("combined tensor and pipeline parallel world size exceeds the InfiniCCL rank limit");
        }
        tp_size_ = static_cast<int>(local_tp_size);
    }

    int tp_size() const noexcept { return tp_size_; }
    int pp_size() const noexcept { return pp_size_; }
    int pp_stage() const noexcept { return pp_stage_; }
    int world_size() const noexcept { return tp_size_ * pp_size_; }

    int world_rank(int local_rank) const {
        if (local_rank < 0 || local_rank >= tp_size_) {
            throw std::out_of_range("tensor parallel rank is out of range");
        }
        return pp_stage_ * tp_size_ + local_rank;
    }

    // Preserve proportional contiguous PP ranges, including stages with zero
    // layers. Compute floor(total * stage / stages) without size_t overflow.
    std::pair<size_t, size_t> layer_range(size_t total_layers) const noexcept {
        const size_t stages = static_cast<size_t>(pp_size_);
        const auto boundary = [=](size_t stage) {
            return (total_layers / stages) * stage
                 + static_cast<size_t>((static_cast<uint64_t>(total_layers % stages) * stage) / stages);
        };
        return {boundary(static_cast<size_t>(pp_stage_)),
                boundary(static_cast<size_t>(pp_stage_) + 1)};
    }

private:
    int tp_size_{1};
    int pp_size_;
    int pp_stage_;
};

} // namespace infinilm::parallel
