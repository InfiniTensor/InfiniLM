#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/scaled_mm_i8.hpp"
#include <infiniop.h>

namespace infinicore::op::scaled_mm_i8_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, I8Gemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, c, a_p, a_s, b_p, b_s;
    std::optional<graph::GraphTensor> bias;
};

void *plan(Tensor c, const Tensor &a_p, const Tensor &a_s, const Tensor &b_p, const Tensor &b_s, std::optional<Tensor> bias) {
    size_t seed = hash_combine(c, a_p, a_s, b_p, b_s);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, I8Gemm,
        seed,
        c->desc(), bias.has_value() ? bias.value()->desc() : nullptr,
        a_p->desc(), a_s->desc(), b_p->desc(), b_s->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, I8Gemm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(c),
        graph::GraphTensor(a_p),
        graph::GraphTensor(a_s),
        graph::GraphTensor(b_p),
        graph::GraphTensor(b_s),
        // bias.has_value() ? bias.value()->desc() : nullptr};
        bias ? std::optional<graph::GraphTensor>(graph::GraphTensor(*bias)) : std::nullopt};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopI8Gemm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->c->data(),
        // planned->bias->data(),
        planned->bias.has_value() ? planned->bias.value()->data() : nullptr,
        planned->a_p->data(),
        planned->a_s->data(),
        planned->b_p->data(),
        planned->b_s->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(I8Gemm, &plan, &run, &cleanup);

} // namespace infinicore::op::scaled_mm_i8_impl::infiniop
