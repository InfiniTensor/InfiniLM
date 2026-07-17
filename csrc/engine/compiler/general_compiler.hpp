#pragma once

#include "paged_compiler.hpp"
#include "piecewise_decode_compiler.hpp"
#include "piecewise_prefill_compiler.hpp"
#include "static_batching_compiler.hpp"

#include <optional>

namespace infinilm::engine {
class GeneralCompiler : public GraphCompiler {
public:
    GeneralCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

    std::optional<infinicore::Tensor> run_native_piecewise_prefill(const InfinilmModel::Input &input);
    std::optional<infinicore::Tensor> run_native_piecewise_decode(const InfinilmModel::Input &input);

    bool native_piecewise_last_prefill_executed() const;

    bool native_piecewise_enabled() const;
    bool native_piecewise_decode_enabled() const;
    const std::vector<size_t> &native_capture_buckets() const;

    void record_graph_hit(bool is_prefill);
    void record_graph_miss(bool is_prefill);
    PagedCompiler::GraphStats graph_stats() const;

private:
    std::unique_ptr<StaticBatchingCompiler> static_batching_compiler_;
    std::unique_ptr<PagedCompiler> paged_compiler_;
    std::unique_ptr<PiecewisePrefillCompiler> piecewise_prefill_compiler_;
    std::unique_ptr<PiecewiseDecodeCompiler> piecewise_decode_compiler_;
};
} // namespace infinilm::engine
