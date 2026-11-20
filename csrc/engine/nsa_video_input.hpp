#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace infinilm::engine::nsa {
// Copied CPU values. Validate before uploads, embedding replacement or KV writes.
inline void validate_video_geometry(std::size_t tokens, std::size_t patches,
                                    const std::vector<std::int64_t> &bounds,
                                    const std::vector<std::int64_t> &grids,
                                    const std::vector<std::size_t> &ranges,
                                    const std::vector<std::int64_t> &positions,
                                    std::size_t position_limit) {
    if (tokens < 2 || tokens > 8192 || bounds.empty() || bounds.size() % 2
        || grids.size() != bounds.size() / 2 * 3 || ranges.size() != bounds.size()
        || positions.size() != 3 * tokens || !position_limit || patches > 4 * tokens) {
        throw std::invalid_argument("NSA video metadata geometry invalid");
    }
    std::int64_t previous = 0;
    std::uint64_t patch_sum = 0;
    for (std::size_t i = 0; i < bounds.size() / 2; ++i) {
        const auto start = bounds[2*i], end = bounds[2*i+1];
        const auto t = grids[3*i], h = grids[3*i+1], w = grids[3*i+2];
        if (t <= 0 || h <= 0 || w <= 0 || t > 32768 || h > 32768 || w > 32768
            || h % 2 || w % 2 || start < previous || start >= end
            || end > static_cast<std::int64_t>(tokens)) {
            throw std::invalid_argument("NSA video grid/bounds invalid");
        }
        const auto count = static_cast<std::uint64_t>(t) * h * w;
        if (count > 4 * tokens || count / 4 != static_cast<std::uint64_t>(end-start)
            || ranges[2*i] != static_cast<std::size_t>(start)
            || ranges[2*i+1] != static_cast<std::size_t>(end)) {
            throw std::invalid_argument("NSA video visual replacement differs from grid");
        }
        patch_sum += count;
        previous = end;
    }
    if (patch_sum != patches) {
        throw std::invalid_argument("NSA video pixel count differs from grids");
    }
    for (const auto p : positions) {
        if (p < 0 || static_cast<std::uint64_t>(p) >= position_limit) {
            throw std::invalid_argument("NSA video position outside rotary cache");
        }
    }
}
} // namespace infinilm::engine::nsa
