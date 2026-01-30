#pragma once

#include "graph_compiler.hpp"

#include <unordered_map>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    PagedCompiler(const std::shared_ptr<InfinilmModel> &model);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    std::vector<size_t> decode_batch_sizes_;

    infinicore::Tensor block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
    };

    std::unordered_map<
        size_t, // num_requests
        CompiledResult>
        compiled_map_decode_;
};
} // namespace infinilm::engine
