#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infinilm::models::minicpmv {

struct MediaInputSlice {
    size_t media_index;
    size_t request_index;
    size_t token_offset;
    size_t token_count;
};

// The processor appends one media bundle per image-bearing request, while
// image_req_ids contains indices into the complete scheduled request batch.
inline std::vector<MediaInputSlice> map_media_to_requests(
    const std::vector<int32_t> &input_offsets,
    size_t packed_token_count,
    const std::vector<size_t> &image_req_ids,
    const std::vector<int32_t> &cached_lengths) {
    if (input_offsets.size() < 2 || input_offsets.front() != 0
        || packed_token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("MiniCPMVModel: invalid packed input_offsets");
    }
    for (size_t i = 1; i < input_offsets.size(); ++i) {
        if (input_offsets[i] < input_offsets[i - 1]) {
            throw std::invalid_argument("MiniCPMVModel: input_offsets must be nondecreasing");
        }
    }
    if (static_cast<size_t>(input_offsets.back()) != packed_token_count) {
        throw std::invalid_argument("MiniCPMVModel: input_offsets must end at packed token count");
    }

    const size_t batch_size = input_offsets.size() - 1;
    if (cached_lengths.size() != batch_size) {
        throw std::invalid_argument("MiniCPMVModel: cached lengths must match request batch");
    }
    for (int32_t length : cached_lengths) {
        if (length < 0) {
            throw std::invalid_argument("MiniCPMVModel: cached lengths must be nonnegative");
        }
    }
    std::vector<bool> seen(batch_size, false);
    std::vector<MediaInputSlice> slices;
    slices.reserve(image_req_ids.size());
    for (size_t media_index = 0; media_index < image_req_ids.size(); ++media_index) {
        const size_t request_index = image_req_ids[media_index];
        if (request_index >= batch_size || seen[request_index]) {
            throw std::invalid_argument("MiniCPMVModel: image_req_ids must identify distinct requests in the batch");
        }
        // The current processor filters cached patches but does not rebase
        // image_bound into the remaining packed input slice.
        if (cached_lengths[request_index] != 0) {
            throw std::invalid_argument("MiniCPMVModel: partially cached media input is not supported");
        }
        seen[request_index] = true;
        const size_t begin = static_cast<size_t>(input_offsets[request_index]);
        const size_t end = static_cast<size_t>(input_offsets[request_index + 1]);
        if (begin == end) {
            throw std::invalid_argument("MiniCPMVModel: image-bearing request has no input tokens");
        }
        slices.push_back({media_index, request_index, begin, end - begin});
    }
    return slices;
}

// Existing MiniCPMV replacement accepts one [1, rows, 2] bound tensor per
// media bundle; the first patch_count rows correspond to its patches.
inline void validate_media_bounds(const std::vector<int64_t> &bounds,
                                  size_t patch_count,
                                  size_t token_count) {
    if (patch_count == 0 || bounds.size() % 2 != 0
        || bounds.size() / 2 < patch_count) {
        throw std::invalid_argument("MiniCPMVModel: image_bound rows do not cover media patches");
    }
    size_t previous_end = 0;
    for (size_t patch = 0; patch < patch_count; ++patch) {
        const int64_t start = bounds[2 * patch];
        const int64_t end = bounds[2 * patch + 1];
        if (start < 0 || end <= start
            || static_cast<uint64_t>(end) > token_count
            || static_cast<size_t>(start) < previous_end) {
            throw std::invalid_argument("MiniCPMVModel: image_bound is outside or overlaps request input tokens");
        }
        previous_end = static_cast<size_t>(end);
    }
    for (size_t row = patch_count; row < bounds.size() / 2; ++row) {
        if (bounds[2 * row] != 0 || bounds[2 * row + 1] != 0) {
            throw std::invalid_argument("MiniCPMVModel: image_bound padding must be zero");
        }
    }
}

} // namespace infinilm::models::minicpmv
