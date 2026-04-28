#include "infinicore/ops/interpolate.hpp"

#include "infiniop/ops/interpolate.h"

#include "../infiniop_impl.hpp"

namespace infinicore::op::interpolate_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Interpolate, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, input;
};

static size_t hash_mode_and_params(const std::string &mode,
                                   int align_corners,
                                   const std::vector<int64_t> &size,
                                   const std::vector<double> &scale_factor) {
    size_t seed = 0;
    hash_combine(seed, mode);
    hash_combine(seed, align_corners);
    hash_combine(seed, size.size());
    for (auto v : size) {
        hash_combine(seed, v);
    }
    hash_combine(seed, scale_factor.size());
    for (auto v : scale_factor) {
        hash_combine(seed, v);
    }
    return seed;
}

void *plan(Tensor out,
           const Tensor &input,
           std::string mode,
           std::vector<int64_t> size,
           std::vector<double> scale_factor,
           int align_corners) {
    const size_t params_hash = hash_mode_and_params(mode, align_corners, size, scale_factor);
    const size_t seed = hash_combine(out, input, params_hash);

    const void *size_ptr = size.empty() ? nullptr : static_cast<const void *>(size.data());
    const void *scale_ptr = scale_factor.empty() ? nullptr : static_cast<const void *>(scale_factor.data());

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Interpolate,
        seed,
        out->desc(),
        input->desc(),
        mode.c_str(),
        const_cast<void *>(size_ptr),
        const_cast<void *>(scale_ptr),
        align_corners);

    INFINIOP_WORKSPACE_TENSOR(workspace, Interpolate, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopInterpolate(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->input->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Interpolate, &plan, &run, &cleanup);

} // namespace infinicore::op::interpolate_impl::infiniop
