#pragma once

#include "chunk_prefill_compiler.hpp"
#include "paged_compiler.hpp"
#include "static_batching_compiler.hpp"

namespace infinilm::engine {
class GeneralCompiler : public GraphCompiler {
public:
    GeneralCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier, bool enable_chunk_prefill_graph = false);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    std::unique_ptr<StaticBatchingCompiler> static_batching_compiler_;
    std::unique_ptr<PagedCompiler> paged_compiler_;
    std::unique_ptr<ChunkPrefillCompiler> chunk_prefill_compiler_;
    bool enable_chunk_prefill_graph_;
};
} // namespace infinilm::engine
