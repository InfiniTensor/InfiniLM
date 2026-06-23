#include "general_compiler.hpp"

#include "../compiled_prefill_flags.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../../utils/agent_debug.hpp"
#include "infinicore/context/context.hpp"

namespace infinilm::engine {

namespace {

void zero_kv_caches_after_compile_() {
    auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    for (auto &kv : kv_cache_vec) {
        if (kv) {
            set_zeros(kv);
        }
    }
    infinicore::context::syncStream();
}

} // namespace
GeneralCompiler::GeneralCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier) : GraphCompiler(model, barrier) {
    static_batching_compiler_ = std::make_unique<StaticBatchingCompiler>(model_, barrier);
    paged_compiler_ = std::make_unique<PagedCompiler>(model_, barrier);
    if (native_piecewise_prefill_enabled()) {
        piecewise_prefill_compiler_ = std::make_unique<PiecewisePrefillCompiler>(model_, barrier);
    }
}

void GeneralCompiler::compile() {
    // #region agent log
    {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "general_compiler.cpp:compile",
            "compile_begin",
            "H2",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + "}",
            "g3b-debug");
    }
    // #endregion
    static_batching_compiler_->compile();
    paged_compiler_->compile();
    if (piecewise_prefill_compiler_ != nullptr) {
        piecewise_prefill_compiler_->compile();
    }
    // #region agent log
    {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "general_compiler.cpp:compile",
            "piecewise_done_pre_zero_kv",
            "H2",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + "}",
            "g3b-debug");
    }
    // #endregion
    zero_kv_caches_after_compile_();
    // #region agent log
    {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "general_compiler.cpp:compile",
            "compile_complete",
            "H2",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + "}",
            "g3b-debug");
    }
    // #endregion
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

std::optional<infinicore::Tensor> GeneralCompiler::run_native_piecewise_prefill(const InfinilmModel::Input &input) {
    if (piecewise_prefill_compiler_ == nullptr) {
        return std::nullopt;
    }
    return piecewise_prefill_compiler_->run_prefill(input);
}

bool GeneralCompiler::native_piecewise_enabled() const {
    return piecewise_prefill_compiler_ != nullptr && piecewise_prefill_compiler_->enabled();
}

const std::vector<size_t> &GeneralCompiler::native_capture_buckets() const {
    static const std::vector<size_t> empty;
    if (piecewise_prefill_compiler_ == nullptr) {
        return empty;
    }
    return piecewise_prefill_compiler_->capture_buckets();
}

void GeneralCompiler::record_graph_hit(bool is_prefill) {
    paged_compiler_->record_graph_hit(is_prefill);
}

void GeneralCompiler::record_graph_miss(bool is_prefill) {
    paged_compiler_->record_graph_miss(is_prefill);
}

PagedCompiler::GraphStats GeneralCompiler::graph_stats() const {
    auto stats = paged_compiler_->graph_stats();
    if (piecewise_prefill_compiler_ != nullptr) {
        stats.piecewise_segment_replays = piecewise_prefill_compiler_->segment_replays();
        stats.piecewise_prefill_hits = piecewise_prefill_compiler_->prefill_hits();
        stats.piecewise_prefill_misses = piecewise_prefill_compiler_->prefill_misses();
    }
    return stats;
}

} // namespace infinilm::engine
