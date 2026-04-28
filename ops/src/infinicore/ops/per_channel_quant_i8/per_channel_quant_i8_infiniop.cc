#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/per_channel_quant_i8.hpp"
#include <infiniop.h>

namespace infinicore::op::per_channel_quant_i8_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PerChannelQuantI8, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, x_packed, x_scale;
};

void *plan(const Tensor &x, Tensor x_packed, Tensor x_scale) {
    size_t seed = hash_combine(x, x_packed, x_scale);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, PerChannelQuantI8,
        seed,
        x_packed->desc(), x_scale->desc(), nullptr, x->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, PerChannelQuantI8, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(x_packed),
        graph::GraphTensor(x_scale)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopPerChannelQuantI8(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x_packed->data(),
        planned->x_scale->data(),
        nullptr,
        planned->x->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(PerChannelQuantI8, &plan, &run, &cleanup);

} // namespace infinicore::op::per_channel_quant_i8_impl::infiniop
