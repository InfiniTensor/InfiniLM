#pragma once

#include "graph_compiler.hpp"

#include <unordered_map>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    std::vector<size_t> decode_batch_sizes_;
    // Experimental fixed-size, single-request Prefill graphs. Unmatched tails
    // keep the ordinary eager path; intermediate graphs do not run the LM head.
    size_t prefill_chunk_size_{0};
    size_t prefill_max_sequence_length_{0};
    void compile_prefill();
    Compiled get_compiled_prefill(const InfinilmModel::Input &input);

    infinicore::Tensor block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
    };

    std::unordered_map<
        size_t, // num_requests
        CompiledResult>
        compiled_map_decode_;
    std::unordered_map<bool, CompiledResult> compiled_map_prefill_;
};
} // namespace infinilm::engine
