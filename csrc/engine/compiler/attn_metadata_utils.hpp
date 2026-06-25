#pragma once

#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../utils/agent_debug.hpp"

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

    if (infinilm::agent_debug::debug_enabled()) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        const int32_t rt_cu0 = infinilm::agent_debug::first_int32(runtime.cu_seqlens.value());
        const int32_t rt_cuN = infinilm::agent_debug::last_int32(runtime.cu_seqlens.value());
        const int32_t cp_cu0 =
            infinilm::agent_debug::first_int32(compiled.cu_seqlens.value()->narrow({{0, 0, cu_len}}));
        const int32_t cp_cuN =
            infinilm::agent_debug::last_int32(compiled.cu_seqlens.value()->narrow({{0, 0, cu_len}}));
        const int32_t meta_cu0 = infinilm::agent_debug::first_int32(meta.cu_seqlens.value());
        const int32_t meta_cuN = infinilm::agent_debug::last_int32(meta.cu_seqlens.value());
        infinilm::agent_debug::log(
            "attn_metadata_utils.hpp:set_attn_metadata_for_varlen_batch",
            "attn_meta_narrow",
            "W1",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"slot_len\":" +
                std::to_string(slot_len) + ",\"rt_cu0\":" + std::to_string(rt_cu0) +
                ",\"rt_cuN\":" + std::to_string(rt_cuN) + ",\"cp_cu0\":" + std::to_string(cp_cu0) +
                ",\"cp_cuN\":" + std::to_string(cp_cuN) + ",\"meta_cu0\":" + std::to_string(meta_cu0) +
                ",\"meta_cuN\":" + std::to_string(meta_cuN) + "}",
            "meta-check");
    }
}

/// Decode replay: same narrow helper; decode batches use fixed width == num_requests.
inline void set_attn_metadata_for_decode_batch(const InfinilmModel::Input &compiled,
                                               const InfinilmModel::Input &runtime) {
    set_attn_metadata_for_varlen_batch(compiled, runtime);
}

} // namespace infinilm::engine::attn_metadata_utils
