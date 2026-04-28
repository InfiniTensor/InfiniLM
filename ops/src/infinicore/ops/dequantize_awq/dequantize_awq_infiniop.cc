#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/dequantize_awq.hpp"
#include <infiniop.h>

namespace infinicore::op::dequantize_awq_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DequantizeAWQ, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, x_packed, x_scale, x_zeros;
};

void *plan(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zeros) {
    size_t seed = hash_combine(x, x_packed, x_scale, x_zeros);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, DequantizeAWQ,
        seed,
        x->desc(), x_packed->desc(), x_scale->desc(), x_zeros->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, DequantizeAWQ, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(x_packed),
        graph::GraphTensor(x_scale),
        graph::GraphTensor(x_zeros)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDequantizeAWQ(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x->data(),
        planned->x_packed->data(),
        planned->x_scale->data(),
        planned->x_zeros->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DequantizeAWQ, &plan, &run, &cleanup);
} // namespace infinicore::op::dequantize_awq_impl::infiniop
