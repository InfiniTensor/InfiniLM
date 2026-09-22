#include "static_batching_compiler.hpp"

namespace infinilm::engine {

StaticBatchingCompiler::StaticBatchingCompiler(
    const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {}

void StaticBatchingCompiler::compile() {
    // Static attention uses host sequence lengths and executes eagerly.
}

StaticBatchingCompiler::Compiled StaticBatchingCompiler::get_compiled(
    const InfinilmModel::Input &) {
    return {nullptr, nullptr};
}

} // namespace infinilm::engine
