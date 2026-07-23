#pragma once

#include "graph_compiler.hpp"

#include <unordered_map>
#include <unordered_set>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    InfinilmModel::Input make_decode_input(size_t batch_size, size_t block_per_req) const;
    void compile_decode(size_t batch_size, size_t block_per_req);

    bool initialized_ = false;
    size_t num_blocks_ = 0;
    size_t block_size_ = 0;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
        infinicore::Tensor padding_total_sequence_lengths;
        infinicore::Tensor padding_request_ids;
        std::unordered_map<size_t, infinicore::Tensor> block_tables_staging;
    };

    std::unordered_map<
        size_t, // static graph batch bucket
        CompiledResult>
        compiled_map_decode_;
    std::unordered_set<size_t> graph_disabled_batches_;
};
} // namespace infinilm::engine
