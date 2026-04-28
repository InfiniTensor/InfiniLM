#include "infinicore/ops/paged_attention.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::paged_attention_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PagedAttention, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, q, k_cache, v_cache, block_tables, cache_lens;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out, const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
           const Tensor &block_tables, const Tensor &cache_lens,
           std::optional<Tensor> alibi_slopes, float scale) {
    size_t seed = hash_combine(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes);
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, PagedAttention,
        seed,
        out->desc(), q->desc(), k_cache->desc(), v_cache->desc(),
        block_tables->desc(), cache_lens->desc(),
        alibi_slopes ? alibi_slopes.value()->desc() : nullptr,
        scale);

    INFINIOP_WORKSPACE_TENSOR(workspace, PagedAttention, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(block_tables),
        graph::GraphTensor(cache_lens),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopPagedAttention(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->out->data(),
            p->q->data(),
            p->k_cache->data(),
            p->v_cache->data(),
            p->block_tables->data(),
            p->cache_lens->data(),
            p->alibi_slopes.has_value() ? p->alibi_slopes.value()->data() : nullptr,
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(PagedAttention, &plan, &run, &cleanup);

} // namespace infinicore::op::paged_attention_impl::infiniop
