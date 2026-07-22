#pragma once

#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"

namespace infinilm::engine::attn_metadata_utils {

inline void set_attn_metadata(const InfinilmModel::Input &input) {
    infinilm::global_state::get_forward_context().attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping,
    };
}

/// Narrow compiled graph buffers to runtime active shapes (RC-2 analog for replay).
inline void set_attn_metadata_for_varlen_batch(const InfinilmModel::Input &compiled,
                                               const InfinilmModel::Input &runtime) {
    const size_t runtime_n_req = runtime.block_tables.value()->size(0);
    const size_t block_per_req = runtime.block_tables.value()->size(1);
    const size_t offset_len = runtime.input_offsets.value()->size(0);
    const size_t cu_len = runtime.cu_seqlens.value()->size(0);
    const size_t slot_len = runtime.slot_mapping.value()->shape()[0];

    auto &meta = infinilm::global_state::get_forward_context().attn_metadata;
    meta.past_sequence_lengths = compiled.past_sequence_lengths.has_value()
                                     ? std::optional<infinicore::Tensor>(
                                           compiled.past_sequence_lengths.value()->narrow({{0, 0, runtime_n_req}}))
                                     : std::nullopt;
    meta.total_sequence_lengths = compiled.total_sequence_lengths.value()->narrow({{0, 0, runtime_n_req}});
    meta.input_offsets = compiled.input_offsets.value()->narrow({{0, 0, offset_len}});
    meta.cu_seqlens = compiled.cu_seqlens.value()->narrow({{0, 0, cu_len}});
    meta.block_tables = compiled.block_tables.value()->narrow({{0, 0, runtime_n_req}, {1, 0, block_per_req}});
    // paged_caching uses slot_mapping.shape[0] as num_tokens (see paged_caching/info.h).
    meta.slot_mapping = compiled.slot_mapping.value()->narrow({{0, 0, slot_len}});
}

/// Decode replay: same narrow helper; decode batches use fixed width == num_requests.
inline void set_attn_metadata_for_decode_batch(const InfinilmModel::Input &compiled,
                                               const InfinilmModel::Input &runtime) {
    set_attn_metadata_for_varlen_batch(compiled, runtime);
}

} // namespace infinilm::engine::attn_metadata_utils
