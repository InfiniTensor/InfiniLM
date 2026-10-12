#include "general_compiler.hpp"

namespace infinilm::engine {
GeneralCompiler::GeneralCompiler(const std::shared_ptr<InfinilmModel> &model,
                                 RankBarrier *barrier,
                                 infinilm::runtime::Runtime &runtime,
                                 infinilm::global_state::ForwardContext &forward_context,
                                 PagedGraphConfig graph_config)
    : GraphCompiler(model, barrier, runtime, forward_context), graph_config_(graph_config) {
    graph_config_.validate();
}

void GeneralCompiler::compile() {
    // A model instance owns exactly one cache strategy. Rebuilding invalidates
    // any graphs that captured the previous cache allocation or layout.
    static_batching_compiler_.reset();
    paged_compiler_.reset();
    if (dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())) {
        static_batching_compiler_ =
            std::make_unique<StaticBatchingCompiler>(model_, barrier_, runtime_, forward_context_);
    static_batching_compiler_->compile();
    } else if (dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        paged_compiler_ = std::make_unique<PagedCompiler>(
            model_, barrier_, runtime_, forward_context_, graph_config_);
    paged_compiler_->compile();
    }
}

GeneralCompiler::Compiled GeneralCompiler::get_compiled(const InfinilmModel::Input &input) {
    if (static_batching_compiler_) {
        return static_batching_compiler_->get_compiled(input);
    }
    if (paged_compiler_) {
        return paged_compiler_->get_compiled(input);
    }
    return {nullptr, nullptr};
}

} // namespace infinilm::engine
