#pragma once

#include "graph_compiler.hpp"

#include <map>
#include <utility>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    void compile_bucket(size_t batch_size, size_t block_per_req);
    bool lazy_capture_ = false;
    std::vector<size_t> decode_batch_sizes_;

    infinicore::Tensor block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
    };

    std::map<
        std::pair<size_t, size_t>, // requests and block-table capacity
        CompiledResult>
        compiled_map_decode_;
};
} // namespace infinilm::engine
