#include "infinicore/ops/embedding.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Embedding);

Embedding::Embedding(Tensor out, const Tensor &input, const Tensor &weight) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input, weight);
}

void Embedding::execute(Tensor out, const Tensor &input, const Tensor &weight) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Embedding, out, input, weight);
}

Tensor embedding(const Tensor &input, // LongTensor of arbitrary shape containing the indices to extract
                 const Tensor &weight // Weight: Embedding matrix of floating point type with shape (V, embedding_dim), where V = maximum index + 1
) {
    auto input_shape = input->shape();
    auto weight_shape = weight->shape();
    auto embedding_dim = weight_shape[1];

    // Assign memory to out variables
    auto output_shape = input_shape;
    output_shape.push_back(embedding_dim);
    Tensor inputs_embeds = Tensor::empty(output_shape, weight->dtype(), weight->device());

    embedding_(inputs_embeds, input, weight);
    return inputs_embeds;
}

void embedding_(Tensor out, const Tensor &input, const Tensor &weight) {
    Embedding::execute(out, input, weight);
}

} // namespace infinicore::op
