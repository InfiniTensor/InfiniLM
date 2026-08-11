#include "infinicore/ops/rotary_embedding.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(RotaryEmbedding);

RotaryEmbedding::RotaryEmbedding(const Tensor &positions,
                                 Tensor query,
                                 std::optional<Tensor> key,
                                 const Tensor &cos_sin_cache,
                                 int64_t head_size,
                                 bool is_neox,
                                 int64_t rope_dim_offset,
                                 bool inverse) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(positions, query, cos_sin_cache);
    if (key) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(query, *key);
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        query->device().type(), positions, query, key, cos_sin_cache,
        head_size, is_neox, rope_dim_offset, inverse);
}

void RotaryEmbedding::execute(const Tensor &positions,
                              Tensor query,
                              std::optional<Tensor> key,
                              const Tensor &cos_sin_cache,
                              int64_t head_size,
                              bool is_neox,
                              int64_t rope_dim_offset,
                              bool inverse) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        RotaryEmbedding, positions, query, key, cos_sin_cache, head_size,
        is_neox, rope_dim_offset, inverse);
}

void rotary_embedding_(const Tensor &positions,
                       Tensor query,
                       std::optional<Tensor> key,
                       const Tensor &cos_sin_cache,
                       int64_t head_size,
                       bool is_neox,
                       int64_t rope_dim_offset,
                       bool inverse) {
    RotaryEmbedding::execute(positions, query, key, cos_sin_cache, head_size,
                             is_neox, rope_dim_offset, inverse);
}

} // namespace infinicore::op
