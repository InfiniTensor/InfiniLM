#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace infinilm::engine {

// Profile selection is independent of eager's legacy table layout policy.
inline void validate_graph_capture_profile(size_t batch, size_t bound, size_t max_batch,
                                           size_t block_size, size_t num_blocks) {
    if (!batch || batch > max_batch || !block_size || !num_blocks
        || num_blocks > std::numeric_limits<size_t>::max() / block_size
        || bound > block_size * num_blocks) {
        throw std::invalid_argument("invalid graph capture profile");
    }
}

inline std::optional<size_t> select_graph_capture_profile(
    const std::vector<std::pair<size_t, size_t>> &profiles, size_t batch,
    size_t actual, size_t pool_tokens) {
    if (!batch || !actual || actual > pool_tokens) {
        return std::nullopt;
    }
    std::optional<size_t> best;
    for (auto [b, bound] : profiles) {
        if (b != batch || (bound && bound < actual)) {
            continue;
        }
        if (!best || (bound && (*best == 0 || bound < *best))) {
            best = bound;
        }
    }
    return best;
}

// Legacy synchronous replay validates active pages without constructing a
// full-pool host table. Its original narrow upload layout remains unchanged.
inline bool valid_graph_page_table(
    const int32_t *lengths, size_t batch, const int32_t *table, size_t source_width,
    size_t target_width, size_t block_size, size_t num_blocks) {
    if (!lengths || !table || !batch || !source_width || !target_width || !block_size
        || batch > std::numeric_limits<size_t>::max() / source_width) {
        return false;
    }
    for (size_t row = 0; row < batch; ++row) {
        if (lengths[row] <= 0) {
            return false;
        }
        const size_t required = 1 + (static_cast<size_t>(lengths[row]) - 1) / block_size;
        if (required > source_width || required > target_width || required > num_blocks) {
            return false;
        }
        for (size_t col = 0; col < required; ++col) {
            const int32_t page = table[row * source_width + col];
            if (page < 0 || static_cast<size_t>(page) >= num_blocks) {
                return false;
            }
        }
    }
    return true;
}

inline std::optional<std::vector<int32_t>> normalize_graph_page_table(
    const int32_t *lengths, size_t batch, const int32_t *table, size_t source_width,
    size_t target_width, size_t block_size, size_t num_blocks) {
    if (!lengths || !table || !batch || !block_size || !source_width || !target_width
        || batch > std::numeric_limits<size_t>::max() / target_width) {
        return std::nullopt;
    }
    std::vector<int32_t> result(batch * target_width, -1);
    for (size_t row = 0; row < batch; ++row) {
        if (lengths[row] <= 0) {
            return std::nullopt;
        }
        const size_t required = 1 + (static_cast<size_t>(lengths[row]) - 1) / block_size;
        if (required > source_width || required > target_width || required > num_blocks) {
            return std::nullopt;
        }
        for (size_t col = 0; col < required; ++col) {
            const int32_t page = table[row * source_width + col];
            if (page < 0 || static_cast<size_t>(page) >= num_blocks) {
                return std::nullopt;
            }
            result[row * target_width + col] = page;
        }
    }
    return result;
}

// One CPU decision for both eager and graph decode. The fixed limited policy
// adds at most four (batch, bound) plans: B=1/2 and bound=256/512 tokens.
// Larger lengths/batches retain the original full-width, bound=0 plan.
struct PagedDecodeBucket {
    size_t batch_size{};
    size_t logical_kv_upper_bound{};
    size_t table_width{};
    std::vector<int32_t> block_table;
};

struct FinitePagedDecodeProfile {
    bool ordinary_kv_prepare{};
    bool paged_attention{};
    bool unquantized_kv{};
    bool short_decode_plan{};
    size_t block_size{};
    size_t num_blocks{};
};

inline bool supports_finite_paged_decode_bucket(const FinitePagedDecodeProfile &profile) {
    return profile.ordinary_kv_prepare && profile.paged_attention
        && profile.unquantized_kv && !profile.short_decode_plan
        && profile.block_size == 256 && profile.num_blocks >= 2;
}

// Paged decode is packed as one row of tokens [1, batch], not [batch, 1].
// The offsets must prove that every request contributes exactly one token.
inline bool is_packed_paged_decode(
    size_t input_rows,
    size_t input_columns,
    const int32_t *input_offsets,
    size_t offset_count,
    size_t batch) {
    if (batch == 0 || batch > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        || input_rows != 1 || input_columns != batch
        || input_offsets == nullptr || offset_count != batch + 1) {
        return false;
    }
    for (size_t row = 0; row <= batch; ++row) {
        if (input_offsets[row] != static_cast<int32_t>(row)) {
            return false;
        }
    }
    return true;
}

inline bool is_packed_paged_decode(
    size_t input_rows,
    size_t input_columns,
    const std::vector<int32_t> &input_offsets,
    size_t batch) {
    return is_packed_paged_decode(input_rows, input_columns,
                                  input_offsets.data(), input_offsets.size(), batch);
}

inline PagedDecodeBucket choose_paged_decode_bucket(
    const int32_t *total_lengths,
    size_t batch,
    const int32_t *source_table,
    size_t source_count,
    size_t source_width,
    size_t block_size,
    size_t num_blocks) {
    if (batch == 0 || total_lengths == nullptr || source_table == nullptr
        || block_size != 256 || num_blocks < 2
        || num_blocks > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        || source_width == 0 || source_width > num_blocks
        || batch > std::numeric_limits<size_t>::max() / source_width
        || source_count != batch * source_width) {
        throw std::invalid_argument("invalid paged decode bucket layout");
    }
    size_t max_total = 0;
    for (size_t row = 0; row < batch; ++row) {
        const int32_t total = total_lengths[row];
        if (total <= 0
            || static_cast<size_t>(total) > num_blocks * block_size) {
            throw std::invalid_argument("paged decode length exceeds cache capacity");
        }
        max_total = std::max(max_total, static_cast<size_t>(total));
    }
    const size_t bound = batch <= 2
                           ? (max_total <= block_size       ? block_size
                              : max_total <= 2 * block_size ? 2 * block_size
                                                            : 0)
                           : 0;
    const size_t width = bound ? bound / block_size : num_blocks;
    if (batch > std::numeric_limits<size_t>::max() / width) {
        throw std::overflow_error("paged decode table size overflows");
    }
    PagedDecodeBucket result{batch, bound, width,
                             std::vector<int32_t>(batch * width, -1)};
    for (size_t row = 0; row < batch; ++row) {
        const size_t required = 1 + (static_cast<size_t>(total_lengths[row]) - 1) / block_size;
        if (required > source_width || required > width) {
            throw std::invalid_argument("paged decode table is shorter than its sequence");
        }
        for (size_t column = 0; column < required; ++column) {
            const int32_t block = source_table[row * source_width + column];
            if (block < 0 || static_cast<size_t>(block) >= num_blocks) {
                throw std::invalid_argument("paged decode block id is out of range");
            }
            result.block_table[row * width + column] = block;
        }
    }
    return result;
}

inline PagedDecodeBucket choose_paged_decode_bucket(
    const std::vector<int32_t> &total_lengths,
    const std::vector<int32_t> &source_table,
    size_t source_width,
    size_t block_size,
    size_t num_blocks) {
    return choose_paged_decode_bucket(
        total_lengths.data(), total_lengths.size(),
        source_table.data(), source_table.size(),
        source_width, block_size, num_blocks);
}

} // namespace infinilm::engine
