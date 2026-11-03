#pragma once
#include "graph_compiler.hpp"
#include <optional>
namespace infinilm::engine {
// Opt-in, bounded exact-shape cache for fresh, single-request Ascend prefill.
class FreshPrefillCompiler : public GraphCompiler {
public:
    FreshPrefillCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
        : GraphCompiler(model, barrier) {}
    void compile() override;
    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    struct Entry {
        InfinilmModel::Input input;
        Compiled compiled;
    };
    std::optional<Entry> entry_;
};
} // namespace infinilm::engine
