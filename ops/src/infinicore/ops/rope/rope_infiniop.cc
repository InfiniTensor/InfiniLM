#include "infinicore/ops/rope.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::rope_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, RoPE, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor x_out;
    graph::GraphTensor x;
    graph::GraphTensor pos;
    graph::GraphTensor sin;
    graph::GraphTensor cos;
};

static infiniopRoPEAlgo_t to_infiniop_algo(infinicore::nn::RoPE::Algo algo) {
    switch (algo) {
    case infinicore::nn::RoPE::Algo::GPT_J:
        return INFINIOP_ROPE_ALGO_GPT_J;
    case infinicore::nn::RoPE::Algo::GPT_NEOX:
        return INFINIOP_ROPE_ALGO_GPT_NEOX;
    default:
        throw std::runtime_error("Unsupported RoPE algorithm");
    }
}

void *plan(Tensor x_out,
           const Tensor &x,
           const Tensor &pos,
           const Tensor &sin,
           const Tensor &cos,
           infinicore::nn::RoPE::Algo algo) {
    auto infiniop_algo = to_infiniop_algo(algo);
    size_t key = hash_combine(x_out, x, pos, sin, cos, static_cast<int>(infiniop_algo));

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, RoPE, key, x_out->desc(),
        x->desc(),
        pos->desc(),
        sin->desc(),
        cos->desc(),
        infiniop_algo);

    INFINIOP_WORKSPACE_TENSOR(workspace, RoPE, descriptor);
    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x_out),
        graph::GraphTensor(x),
        graph::GraphTensor(pos),
        graph::GraphTensor(sin),
        graph::GraphTensor(cos)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopRoPE(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->x_out->data(),
            p->x->data(),
            p->pos->data(),
            p->sin->data(),
            p->cos->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(RoPE, &plan, &run, &cleanup);

} // namespace infinicore::op::rope_impl::infiniop
