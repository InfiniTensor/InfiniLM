#pragma once

#include "graph_compiler.hpp"

namespace infinilm::engine {
class StaticBatchingCompiler : public GraphCompiler {
public:
    StaticBatchingCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);
    void compile() override;
    Compiled get_compiled(const InfinilmModel::Input &input) override;
};
} // namespace infinilm::engine
