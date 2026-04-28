#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/scaled_mm_w4a16_gptq_qy.hpp"
#include <infiniop.h>

namespace infinicore::op::scaled_mm_w4a16_gptq_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, GptqQyblasGemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, in, qweight, scales, qzeros;
    int64_t quant_type, bit;
};

void *plan(Tensor out, const Tensor &in, const Tensor &qweight, const Tensor &scales, const Tensor &qzeros, int64_t quant_type, int64_t bit) {
    size_t seed = hash_combine(out, in, qweight, scales, qzeros);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, GptqQyblasGemm,
        seed,
        out->desc(), in->desc(), qweight->desc(), scales->desc(), qzeros->desc());
    INFINIOP_WORKSPACE_TENSOR(workspace, GptqQyblasGemm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(in),
        graph::GraphTensor(qweight),
        graph::GraphTensor(scales),
        graph::GraphTensor(qzeros),
        quant_type, bit};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopGptqQyblasGemm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->in->data(),
        planned->qweight->data(),
        planned->scales->data(),
        planned->qzeros->data(),
        planned->quant_type, planned->bit,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(GptqQyblasGemm, &plan, &run, &cleanup);

} // namespace infinicore::op::scaled_mm_w4a16_gptq_impl::infiniop
