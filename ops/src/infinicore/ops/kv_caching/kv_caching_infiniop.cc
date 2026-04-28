#include "../infiniop_impl.hpp"
#include "infinicore/ops/kv_caching.hpp"

namespace infinicore::op::kv_caching_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, KVCaching, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, k_cache, v_cache, k, v, past_kv_lengths;
};

void *plan(Tensor k_cache,
           Tensor v_cache,
           const Tensor &k,
           const Tensor &v,
           const Tensor &past_kv_lengths) {
    size_t seed = hash_combine(k_cache, v_cache, k, v, past_kv_lengths);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, KVCaching,
        seed, k_cache->desc(), v_cache->desc(),
        k->desc(), v->desc(), past_kv_lengths->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, KVCaching, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(past_kv_lengths)};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopKVCaching(
        planned->descriptor->desc,
        nullptr, 0,
        planned->k_cache->data(),
        planned->v_cache->data(),
        planned->k->data(),
        planned->v->data(),
        planned->past_kv_lengths->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(KVCaching, &plan, &run, cleanup);

} // namespace infinicore::op::kv_caching_impl::infiniop
