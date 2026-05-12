#include "general_compiler.hpp"

namespace infinilm::engine {
GeneralCompiler::GeneralCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier, bool enable_chunk_prefill_graph)
    : GraphCompiler(model, barrier), enable_chunk_prefill_graph_(enable_chunk_prefill_graph) {
    static_batching_compiler_ = std::make_unique<StaticBatchingCompiler>(model_, barrier);
    chunk_prefill_compiler_ = std::make_unique<ChunkPrefillCompiler>(model_, barrier);
    paged_compiler_ = std::make_unique<PagedCompiler>(model_, barrier);
}

void GeneralCompiler::compile() {
    static_batching_compiler_->compile();
    if (enable_chunk_prefill_graph_) {
        chunk_prefill_compiler_->compile();
    }
    paged_compiler_->compile();
}

GeneralCompiler::Compiled GeneralCompiler::get_compiled(const InfinilmModel::Input &input) {
    GeneralCompiler::Compiled result = {nullptr, nullptr};

    // try each compiler, return the first valid result
    result = static_batching_compiler_.get()->get_compiled(input);
    if (std::get<0>(result) != nullptr && std::get<1>(result) != nullptr) {
        return result;
    }
    // chunk-prefill must be checked before decode (decode would also match if chunk_size==1)
    result = chunk_prefill_compiler_.get()->get_compiled(input);
    if (std::get<0>(result) != nullptr && std::get<1>(result) != nullptr) {
        return result;
    }
    result = paged_compiler_.get()->get_compiled(input);
    return result;
}

} // namespace infinilm::engine
