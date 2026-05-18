#pragma once

#include "graph_compiler.hpp"

#include <unordered_map>

namespace infinilm::engine {
class ChunkPrefillCompiler : public GraphCompiler {
public:
    ChunkPrefillCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

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

    std::vector<size_t> chunk_sizes_;
    std::vector<size_t> prefill_batch_sizes_;

    infinicore::Tensor block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
    };

    // Key: (batch_size, chunk_size)
    std::unordered_map<
        std::tuple<size_t, size_t>,
        CompiledResult,
        TupleHash>
        compiled_map_prefill_;
};
} // namespace infinilm::engine
