#include "../infiniop_impl.hpp"
#include "infinicore/ops/embedding.hpp"

namespace infinicore::op::embedding_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Embedding, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor out, input, weight;
};

void *plan(Tensor out, const Tensor &input, const Tensor &weight) {
    size_t seed = hash_combine(out, input, weight);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Embedding,
        seed, out->desc(), input->desc(), weight->desc());

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(weight)};

    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopEmbedding(
        planned->descriptor->desc,
        planned->out->data(), planned->input->data(), planned->weight->data(), context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Embedding, &plan, &run, cleanup);

} // namespace infinicore::op::embedding_impl::infiniop
