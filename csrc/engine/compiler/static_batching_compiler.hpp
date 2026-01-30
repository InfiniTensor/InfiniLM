#pragma once

#include "graph_compiler.hpp"

#include <unordered_map>

namespace infinilm::engine {
class StaticBatchingCompiler : public GraphCompiler {
public:
    StaticBatchingCompiler(const std::shared_ptr<InfinilmModel> &model);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

private:
    struct TupleHash {
        size_t operator()(const std::tuple<size_t, size_t> &t) const noexcept {
            auto h1 = std::hash<size_t>{}(std::get<0>(t));
            auto h2 = std::hash<size_t>{}(std::get<1>(t));
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
    };

    std::unordered_map<
        std::tuple<size_t, size_t>, // (batch_size, seq_len)
        CompiledResult,
        TupleHash>
        compiled_map_;
};
} // namespace infinilm::engine
