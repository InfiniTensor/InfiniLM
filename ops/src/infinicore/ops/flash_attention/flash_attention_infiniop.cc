#include "../../utils.hpp"
#include "../infiniop_impl.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/flash_attention.hpp"
#include <infiniop.h>

namespace infinicore::op::flash_attention_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, FlashAttention, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, q, k, v, total_kv_len;
    float scale;
    bool is_causal;
};

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &total_kv_len, float scale, bool is_causal) {
    size_t seed = hash_combine(out, q, k, v, total_kv_len, scale, is_causal);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, FlashAttention,
        seed, out->desc(), q->desc(), k->desc(), v->desc(), total_kv_len->desc(), scale, is_causal);

    INFINIOP_WORKSPACE_TENSOR(workspace, FlashAttention, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(total_kv_len), scale, is_causal};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopFlashAttention(
        planned->descriptor->desc, planned->workspace->data(), planned->workspace->numel(),
        planned->out->data(), planned->q->data(), planned->k->data(), planned->v->data(), planned->total_kv_len->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(FlashAttention, &plan, &run, &cleanup);

} // namespace infinicore::op::flash_attention_impl::infiniop
