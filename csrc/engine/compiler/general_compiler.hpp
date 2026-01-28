#pragma once

#include "paged_compiler.hpp"
#include "static_batching_compiler.hpp"

namespace infinilm::engine {
class GeneralCompiler : public GraphCompiler {
public:
    GeneralCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    std::unique_ptr<StaticBatchingCompiler> static_batching_compiler_;
    std::unique_ptr<PagedCompiler> paged_compiler_;
};
} // namespace infinilm::engine
