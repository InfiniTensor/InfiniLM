#pragma once

#include "graph_compiler.hpp"

#include <limits>
#include <unordered_map>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

    std::pair<std::shared_ptr<infinicore::graph::Graph>, infinicore::Tensor>
    get_sampling_compiled(size_t batch_size) override;

private:
    std::vector<size_t> decode_batch_sizes_;

    infinicore::Tensor block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
        // Small per-step inputs live as views into two contiguous device
        // buffers so each replay needs only two H2D copies. Host staging is
        // reused across steps.
        infinicore::Tensor pack_i64; // input_ids | position_ids | slot_mapping
        infinicore::Tensor pack_i32; // total_seq_lens | input_offsets | cu_seqlens
        std::vector<int64_t> stage_i64;
        std::vector<int32_t> stage_i32;
        size_t position_id_axes = 1;
        // Captured per-request argmax over the decode graph's logits blob,
        // replayed for greedy batches to avoid ~3 kernel launches per request.
        std::shared_ptr<infinicore::graph::Graph> sampling_graph;
        infinicore::Tensor sampling_out;
    };

    // Decode-kernel ctx routing: when decode_fa_ctx_threshold() is finite,
    // each batch size gets a second capture whose attention layers recorded
    // the FA kvcache kernel (routed by the fake max_sequence_length used at
    // capture time). get_compiled() picks the variant by the runtime batch's
    // max context length, which it already has on the host.
    struct DecodeVariants {
        CompiledResult short_ctx;
        CompiledResult long_ctx;
        bool has_long = false;
        // Set by get_compiled(); get_sampling_compiled() serves the sampling
        // graph of the variant it most recently handed out.
        CompiledResult *last_served = nullptr;
    };
    std::unordered_map<size_t, DecodeVariants> compiled_map_decode_;
    size_t decode_ctx_threshold_ = std::numeric_limits<size_t>::max();
};
} // namespace infinilm::engine
