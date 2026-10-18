#pragma once
#include <stdexcept>

#include "../engine/nsa_owner_input.hpp"
#include "../models/infinilm_model.hpp"

namespace infinilm::runtime {
class Runtime;
}

namespace infinilm::global_state {

// Proven once from host decode metadata before a static forward. The generic
// static attention path may only use one shared length/cache position when
// both properties hold; fused cache attention consumes the device row table.
struct StaticDecodeHostDescription {
    bool identity_rows{true};
    bool uniform_lengths{true};
};

// Validated once from CPU metadata before an eager rectangular prefill.  A
// single request may address one row of a larger physical static cache.
struct StaticPrefillHostDescription {
    size_t batch_size{};
    size_t query_length{};
    size_t cache_capacity{};
    size_t page_size{};
    size_t physical_row{}; // Used only when batch_size == 1.
    size_t past{};
    size_t total{};
};

struct AttentionMetadata {
    /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
    std::optional<infinilm::runtime::Tensor> past_sequence_lengths;
    /// ToTal Lengths for each request sequence, of shape `[num_requests]`.
    std::optional<infinilm::runtime::Tensor> total_sequence_lengths;
    /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
    std::optional<infinilm::runtime::Tensor> input_offsets;
    /// Cumulative total sequence lengths for each request, of shape `[num_requests + 1]`.
    std::optional<infinilm::runtime::Tensor> cu_seqlens;
    /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
    std::optional<infinilm::runtime::Tensor> block_tables;
    /// Slot ids for each token `[seq]`. Used for paged cache.
    std::optional<infinilm::runtime::Tensor> slot_mapping;
    /// Maximum query length in the current batch.
    size_t max_query_length{0};
    /// Maximum total sequence length in the current batch.
    size_t max_sequence_length{0};
    /// Zero preserves the physical-table scheduling bound; nonzero is the
    /// validated per-forward decode planning upper bound in tokens.
    size_t logical_kv_upper_bound{0};
    /// Element 0 of past_sequence_lengths, snapshotted for static eager attention.
    std::optional<size_t> first_past_sequence_length;
    /// Element 0 of total_sequence_lengths, snapshotted for static eager attention.
    std::optional<size_t> first_total_sequence_length;
    /// Present only after one worker-side validation of static decode inputs.
    std::optional<StaticDecodeHostDescription> static_decode_host;
    std::optional<StaticPrefillHostDescription> static_prefill_host;

    AttentionMetadata() = default;

    AttentionMetadata(std::optional<infinilm::runtime::Tensor> past_sequence_lengths,
                      std::optional<infinilm::runtime::Tensor> total_sequence_lengths,
                      std::optional<infinilm::runtime::Tensor> input_offsets,
                      std::optional<infinilm::runtime::Tensor> cu_seqlens,
                      std::optional<infinilm::runtime::Tensor> block_tables,
                      std::optional<infinilm::runtime::Tensor> slot_mapping,
                      size_t max_query_length = 0,
                      size_t max_sequence_length = 0,
                      size_t logical_kv_upper_bound = 0) : past_sequence_lengths(past_sequence_lengths),
                                                           total_sequence_lengths(total_sequence_lengths),
                                                           input_offsets(input_offsets),
                                                           cu_seqlens(cu_seqlens),
                                                           block_tables(block_tables),
                                                           slot_mapping(slot_mapping),
                                                           max_query_length(max_query_length),
                                                           max_sequence_length(max_sequence_length),
                                                           logical_kv_upper_bound(logical_kv_upper_bound) {}

    AttentionMetadata(const infinilm::InfinilmModel::Input &input) : AttentionMetadata(input.past_sequence_lengths,
                                                                                       input.total_sequence_lengths,
                                                                                       input.input_offsets,
                                                                                       input.cu_seqlens,
                                                                                       input.block_tables,
                                                                                       input.slot_mapping,
                                                                                       0,
                                                                                       0,
                                                                                       input.logical_kv_upper_bound) {}
};

struct MultiModalMetadata {
    std::optional<std::vector<size_t>> image_req_ids;
    // Flattened [start, end) token ranges in the current packed language sequence.
    std::optional<std::vector<size_t>> visual_token_ranges;
};

struct MambaMetadata {
    /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
    std::optional<infinilm::runtime::Tensor> input_offsets;
    /// State cache indices read at the start of each request forward.
    std::optional<infinilm::runtime::Tensor> init_state_indices;
    /// State cache indices written with the final state of each request forward.
    std::optional<infinilm::runtime::Tensor> final_state_indices;
};

struct ForwardContext {
    infinilm::runtime::Runtime &require_runtime() const {
        if (!runtime) {
            throw std::logic_error("ForwardContext requires its worker Runtime");
        }
        return *runtime;
    }
    // Worker-owned; the context is used only while that worker is alive.
    infinilm::runtime::Runtime *runtime{nullptr};
    std::shared_ptr<const engine::nsa::Proof> nsa_owner;
    AttentionMetadata attn_metadata;
    MambaMetadata mamba_metadata;
    MultiModalMetadata mm_metadata;
    std::vector<infinilm::runtime::Tensor> kv_cache_vec;
    std::vector<infinilm::runtime::Tensor> conv_state_vec;
    std::vector<infinilm::runtime::Tensor> ssm_state_vec;
};

} // namespace infinilm::global_state
