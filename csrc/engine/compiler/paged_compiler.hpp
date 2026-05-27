#pragma once

#include "graph_compiler.hpp"

#include <unordered_map>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    struct GraphStats {
        size_t prefill_graph_hits{0};
        size_t prefill_graph_misses{0};
        size_t decode_graph_hits{0};
        size_t decode_graph_misses{0};
    };

    PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

    void record_graph_hit(bool is_prefill);
    void record_graph_miss(bool is_prefill);
    GraphStats graph_stats() const;

private:
    std::vector<size_t> decode_batch_sizes_;
    std::vector<size_t> prefill_seq_buckets_{4096};

    infinicore::Tensor block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
    };

    std::unordered_map<
        size_t, // num_requests
        CompiledResult>
        compiled_map_decode_;

    std::unordered_map<
        size_t, // prefill sequence bucket length
        CompiledResult>
        compiled_map_prefill_;

    size_t prefill_graph_hits_{0};
    size_t prefill_graph_misses_{0};
    size_t decode_graph_hits_{0};
    size_t decode_graph_misses_{0};
};
} // namespace infinilm::engine
