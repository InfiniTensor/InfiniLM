#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace infinilm::engine {

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
    bool nvidia{};
    bool native_flash_decode{};
    bool tensor_parallel_qualified{};
    bool ordinary_kv_prepare{};
    bool unquantized{};
    bool half_dtype{};
    bool short_decode_plan{};
    size_t head_dim{};
    size_t block_size{};
    size_t num_blocks{};
};

// TP2 finite buckets currently require the Qwen3 BF16 geometry below.
// MoE and custom families retain their original decode plans.
inline bool supports_qwen3_tp2_finite_decode(
    int tp_size,int pp_size,std::string_view model_type,size_t hidden_size,
    size_t attention_heads,size_t kv_heads,size_t depth,size_t position_axes,
    bool bfloat16) {
    return tp_size==2 && pp_size==1 && model_type=="qwen3" &&
        hidden_size==1024 && attention_heads==16 && kv_heads==8 &&
        depth==28 && position_axes==1 && bfloat16;
}

inline bool supports_finite_paged_decode_bucket(const FinitePagedDecodeProfile &profile) {
    return profile.nvidia && profile.native_flash_decode
        && profile.tensor_parallel_qualified && profile.ordinary_kv_prepare
        && profile.unquantized && profile.half_dtype
        && !profile.short_decode_plan && profile.head_dim == 128
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
        ? (max_total <= block_size ? block_size
           : max_total <= 2 * block_size ? 2 * block_size : 0)
        : 0;
    const size_t width = bound ? bound / block_size : num_blocks;
    if (batch > std::numeric_limits<size_t>::max() / width) {
        throw std::overflow_error("paged decode table size overflows");
    }
    PagedDecodeBucket result{batch, bound, width,
                             std::vector<int32_t>(batch * width, -1)};
    for (size_t row = 0; row < batch; ++row) {
        const size_t required = 1 + (static_cast<size_t>(total_lengths[row]) - 1)
                                      / block_size;
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
