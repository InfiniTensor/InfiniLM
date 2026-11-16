#pragma once

#include "runtime/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace infinilm::engine {

struct DeepSeekV4AttentionMetadata {
    runtime::Tensor swa_indices;
    runtime::Tensor swa_topk_lengths;
    runtime::Tensor raw_out_loc;
    runtime::Tensor page_table;
    runtime::Tensor c4_out_loc;
    runtime::Tensor c4_positions;
    runtime::Tensor c4_topk_lengths_raw;
    runtime::Tensor c4_sparse_topk_lengths;
    runtime::Tensor c128_out_loc;
    runtime::Tensor c128_positions;
    runtime::Tensor c128_page_indices;
    runtime::Tensor c128_topk_lengths_clamp1;
    runtime::Tensor c4_compress_write_loc;
    runtime::Tensor c4_compress_extra_loc;
    runtime::Tensor c128_compress_write_loc;
};

// Inputs and returned tensors are contiguous host metadata. Device copies
// belong to the worker Runtime that will submit attention.
DeepSeekV4AttentionMetadata build_deepseek_v4_attention_metadata(
    const runtime::Tensor &block_tables,
    const runtime::Tensor &slot_mapping,
    const runtime::Tensor &position_ids,
    const runtime::Tensor &input_offsets,
    const std::optional<std::vector<int64_t>> &full_to_swa_block_ids,
    size_t block_size);

struct DeepSeekV4CompressionPlan {
    runtime::Tensor sources;
    runtime::Tensor state_slots;
};

// Build the device compressor's source map on the host. Rows from the current
// projection take precedence over historical ring rows. Only the last writer
// to each ring slot is committed after pooling.
DeepSeekV4CompressionPlan build_deepseek_v4_compression_plan(
    const DeepSeekV4AttentionMetadata &metadata,
    const runtime::Tensor &position_ids,
    const runtime::Tensor &input_offsets,
    size_t ratio);

} // namespace infinilm::engine
