#include "infinicore/ops/deepseek_moe.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekMoe);

namespace {

void check_weights(const std::vector<Tensor> &gate_weights,
                   const std::vector<Tensor> &up_weights,
                   const std::vector<Tensor> &down_weights,
                   size_t num_experts) {
    if (gate_weights.size() != num_experts || up_weights.size() != num_experts || down_weights.size() != num_experts) {
        throw std::runtime_error("DeepseekMoe: expert weight vector size mismatch");
    }
}

} // namespace

DeepseekMoe::DeepseekMoe(Tensor out,
                         const Tensor &hidden,
                         const Tensor &topk_indices,
                         const Tensor &topk_weights,
                         const std::vector<Tensor> &gate_weights,
                         const std::vector<Tensor> &up_weights,
                         const std::vector<Tensor> &down_weights,
                         size_t intermediate_size,
                         size_t num_experts) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, hidden, topk_indices, topk_weights);
    check_weights(gate_weights, up_weights, down_weights, num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, gate_weights[i], up_weights[i], down_weights[i]);
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().type(),
                                 out, hidden, topk_indices, topk_weights,
                                 gate_weights, up_weights, down_weights,
                                 intermediate_size, num_experts);
}

void DeepseekMoe::execute(Tensor out,
                          const Tensor &hidden,
                          const Tensor &topk_indices,
                          const Tensor &topk_weights,
                          const std::vector<Tensor> &gate_weights,
                          const std::vector<Tensor> &up_weights,
                          const std::vector<Tensor> &down_weights,
                          size_t intermediate_size,
                          size_t num_experts) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekMoe,
        out, hidden, topk_indices, topk_weights,
        gate_weights, up_weights, down_weights,
        intermediate_size, num_experts);
}

void deepseek_moe_(Tensor out,
                   const Tensor &hidden,
                   const Tensor &topk_indices,
                   const Tensor &topk_weights,
                   const std::vector<Tensor> &gate_weights,
                   const std::vector<Tensor> &up_weights,
                   const std::vector<Tensor> &down_weights,
                   size_t intermediate_size,
                   size_t num_experts) {
    DeepseekMoe::execute(out, hidden, topk_indices, topk_weights, gate_weights, up_weights, down_weights, intermediate_size, num_experts);
}

Tensor deepseek_moe(const Tensor &hidden,
                    const Tensor &topk_indices,
                    const Tensor &topk_weights,
                    const std::vector<Tensor> &gate_weights,
                    const std::vector<Tensor> &up_weights,
                    const std::vector<Tensor> &down_weights,
                    size_t intermediate_size,
                    size_t num_experts) {
    auto out = Tensor::empty(hidden->shape(), hidden->dtype(), hidden->device());
    deepseek_moe_(out, hidden, topk_indices, topk_weights, gate_weights, up_weights, down_weights, intermediate_size, num_experts);
    return out;
}

} // namespace infinicore::op
