#include "infinicore/ops/rope.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/rotary_embedding_infinilm.h"

#include <stdexcept>

namespace infinicore::op::rope_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    TensorMeta x_out, x, pos, sin, cos;
    graph::GraphTensor x_out_tensor, x_tensor, pos_tensor, sin_tensor, cos_tensor;
    bool is_neox;
};

bool toInfiniOpsIsNeox(infinicore::nn::RoPE::Algo algo) {
    switch (algo) {
    case infinicore::nn::RoPE::Algo::GPT_J:
        return true;
    case infinicore::nn::RoPE::Algo::GPT_NEOX:
        return false;
    default:
        throw std::runtime_error("Unsupported RoPE algorithm");
    }
}
} // namespace

void *plan(Tensor x_out,
           const Tensor &x,
           const Tensor &pos,
           const Tensor &sin,
           const Tensor &cos,
           infinicore::nn::RoPE::Algo algo) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(x_out->device().type()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x_out, x, pos, sin, cos);
    return new PlannedMeta{
        TensorMeta(x_out), TensorMeta(x), TensorMeta(pos), TensorMeta(sin), TensorMeta(cos),
        graph::GraphTensor(x_out), graph::GraphTensor(x), graph::GraphTensor(pos), graph::GraphTensor(sin), graph::GraphTensor(cos),
        toInfiniOpsIsNeox(algo)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::RotaryEmbeddingInfinilm::Call(
        handle,
        config,
        planned->x.tensor(planned->x_tensor),
        planned->pos.tensor(planned->pos_tensor),
        planned->sin.tensor(planned->sin_tensor),
        planned->cos.tensor(planned->cos_tensor),
        planned->is_neox,
        planned->x_out.tensor(planned->x_out_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(RoPE::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(RoPE::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(RoPE::cleanup_dispatcher(), &cleanup);
    return true;
}();
} // namespace infinicore::op::rope_impl::infiniops
#endif
