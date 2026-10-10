#include "infinicore/ops/moe_topk_sigmoid.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

namespace infinicore::op::moe_topk_sigmoid_impl::aten {
namespace {

struct PlannedMeta {
    Tensor topk_weights;
    Tensor topk_indices;
    Tensor gating_output;
    Tensor correction_bias;
    bool renormalize;
};

void *plan(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &gating_output,
           const Tensor &correction_bias,
           const bool renormalize) {
    INFINICORE_ASSERT(topk_weights->device().type() == Device::Type::kQy);
    return new PlannedMeta{
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        renormalize};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    using namespace ::infinicore::adaptor;
    set_aten_stream_to_infinicore();

    // Match InfiniOP moe_topk_sigmoid: rank with sigmoid(+bias), emit debiased
    // sigmoid probs as weights (subtract bias after top-k), then optionally renormalize.
    auto gating = to_aten_tensor(planned->gating_output);
    auto probs = gating.sigmoid().to(at::kFloat);
    at::Tensor scores = probs;
    at::Tensor bias;
    if (planned->correction_bias) {
        bias = to_aten_tensor(planned->correction_bias).to(at::kFloat);
        scores = probs + bias;
    }

    const auto topk = static_cast<int64_t>(planned->topk_weights->size(1));
    auto topk_out = at::topk(scores, topk, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
    auto values = std::get<0>(topk_out);
    auto indices = std::get<1>(topk_out);

    if (planned->correction_bias) {
        // Debias selected scores back to pure sigmoid probs (matches CUDA kernel).
        values = values - bias.unsqueeze(0).expand_as(scores).gather(/*dim=*/1, indices);
    }

    if (planned->renormalize) {
        values = values / values.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(1e-20);
    }

    to_aten_tensor(planned->topk_weights).copy_(values.to(at::kFloat));
    if (planned->topk_indices->dtype() == DataType::kInt32) {
        to_aten_tensor(planned->topk_indices).copy_(indices.to(at::kInt));
    } else {
        to_aten_tensor(planned->topk_indices).copy_(indices.to(at::kLong));
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    MoeTopkSigmoid::plan_dispatcher().registerDevice(Device::Type::kQy, &plan, true);
    MoeTopkSigmoid::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MoeTopkSigmoid::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::moe_topk_sigmoid_impl::aten
#endif
