#include "general_compiler.hpp"

namespace infinilm::engine {
GeneralCompiler::GeneralCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier) : GraphCompiler(model, barrier) {
    static_batching_compiler_ = std::make_unique<StaticBatchingCompiler>(model_, barrier);
    paged_compiler_ = std::make_unique<PagedCompiler>(model_, barrier);
}

void GeneralCompiler::compile() {
    static_batching_compiler_->compile();
    paged_compiler_->compile();
}

GeneralCompiler::Compiled GeneralCompiler::get_compiled(const InfinilmModel::Input &input) {
    GeneralCompiler::Compiled result = {nullptr, nullptr};

    // try each compiler, return the first valid result
    result = static_batching_compiler_.get()->get_compiled(input);
    if (std::get<0>(result) != nullptr && std::get<1>(result) != nullptr) {
        return result;
    }
    result = paged_compiler_.get()->get_compiled(input);
    return result;
}

} // namespace infinilm::engine
