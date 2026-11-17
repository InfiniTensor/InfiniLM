#include "infinicore/ops/causal_softmax.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ATen.h>

#include <cmath>
#include <limits>

namespace infinicore::op::causal_softmax_impl::aten {
namespace {

struct PlannedMeta {
    Tensor output;
    Tensor input;
};

void *plan(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT(output->device().type() == Device::Type::kQy);
    return new PlannedMeta{output, input};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    using namespace ::infinicore::adaptor;
    set_aten_stream_to_infinicore();

    auto input = to_aten_tensor(p->input);
    auto output = to_aten_tensor(p->output);
    INFINICORE_ASSERT(input.dim() == 2 || input.dim() == 3);

    const auto seq_len = input.size(-2);
    const auto total_seq_len = input.size(-1);
    INFINICORE_ASSERT(seq_len <= total_seq_len);

    const bool reuses_input = input.is_same(output);
    const bool needs_mask = seq_len != 1 || !reuses_input;

    at::Tensor scores;
    if (needs_mask) {
        const int64_t score_diagonal = total_seq_len - seq_len;
        const int64_t mask_diagonal = total_seq_len - seq_len + 1;
        // Match InfiniOps causal_softmax: tril(scores) + triu(-inf mask).
        auto tril_scores = at::tril(input, score_diagonal);
        auto mask = at::full({seq_len, total_seq_len},
                             -std::numeric_limits<float>::infinity(),
                             input.options());
        mask = at::triu(mask, mask_diagonal);
        // Broadcast mask over leading dims if present.
        scores = tril_scores + mask;
    } else {
        scores = input;
    }

    auto probs = at::softmax(scores, /*dim=*/-1);
    output.copy_(probs);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    CausalSoftmax::plan_dispatcher().registerDevice(Device::Type::kQy, &plan, true);
    CausalSoftmax::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    CausalSoftmax::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::causal_softmax_impl::aten
#endif
