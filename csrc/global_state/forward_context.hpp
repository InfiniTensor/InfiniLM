#pragma once

#include "../models/infinilm_model.hpp"

namespace infinilm::global_state {

struct AttentionMetadata {
    /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
    std::optional<infinicore::Tensor> position_ids;
    /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
    std::optional<infinicore::Tensor> past_sequence_lengths;
    /// ToTal Lengths for each request sequence, of shape `[num_requests]`.
    std::optional<infinicore::Tensor> total_sequence_lengths;
    /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
    std::optional<infinicore::Tensor> input_offsets;
    /// Cumulative total sequence lengths for each request, of shape `[num_requests + 1]`.
    std::optional<infinicore::Tensor> cu_seqlens;
    /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
    std::optional<infinicore::Tensor> block_tables;
    /// Slot ids for each token `[seq]`. Used for paged cache.
    std::optional<infinicore::Tensor> slot_mapping;

    AttentionMetadata() = default;

    AttentionMetadata(std::optional<infinicore::Tensor> position_ids,
                      std::optional<infinicore::Tensor> past_sequence_lengths,
                      std::optional<infinicore::Tensor> total_sequence_lengths,
                      std::optional<infinicore::Tensor> input_offsets,
                      std::optional<infinicore::Tensor> cu_seqlens,
                      std::optional<infinicore::Tensor> block_tables,
                      std::optional<infinicore::Tensor> slot_mapping) : position_ids(position_ids),
                                                                        past_sequence_lengths(past_sequence_lengths),
                                                                        total_sequence_lengths(total_sequence_lengths),
                                                                        input_offsets(input_offsets),
                                                                        cu_seqlens(cu_seqlens),
                                                                        block_tables(block_tables),
                                                                        slot_mapping(slot_mapping) {}

    AttentionMetadata(const infinilm::InfinilmModel::Input &input) : AttentionMetadata(input.position_ids,
                                                                                       input.past_sequence_lengths,
                                                                                       input.total_sequence_lengths,
                                                                                       input.input_offsets,
                                                                                       input.cu_seqlens,
                                                                                       input.block_tables,
                                                                                       input.slot_mapping) {}
};

struct ForwardContext {
    AttentionMetadata attn_metadata;
    std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> kv_cache_vec;
};

void initialize_forward_context(ForwardContext &forward_context);

ForwardContext &get_forward_context();

} // namespace infinilm::global_state
