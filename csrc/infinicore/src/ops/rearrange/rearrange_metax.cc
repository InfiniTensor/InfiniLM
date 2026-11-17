#include "infinicore/ops/rearrange.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace infinicore::op::rearrange_impl::metax {
namespace {

struct PlannedMeta {
    Tensor y;
    Tensor x;
};

void *plan(Tensor y, const Tensor &x) {
    INFINICORE_ASSERT(y->device().type() == Device::Type::kMetax);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    INFINICORE_ASSERT(y->dtype() == x->dtype());
    INFINICORE_ASSERT(y->shape() == x->shape());
    return new PlannedMeta{y, x};
}

void run(void *planned_meta) {
    auto *meta = reinterpret_cast<PlannedMeta *>(planned_meta);
    auto y = meta->y;
    auto x = meta->x;
    const size_t nbytes = y->nbytes();
    if (nbytes == 0 || y->data() == x->data()) {
        return;
    }
    context::setDevice(y->device());
    if (y->is_contiguous() && x->is_contiguous()) {
        context::memcpyD2D(y->data(), x->data(), nbytes);
        return;
    }
    // Non-contiguous: stage through host (weight-load rare path).
    auto host = Tensor::empty(y->shape(), y->dtype(), Device{Device::Type::kCpu});
    // src device -> host
    if (x->is_contiguous()) {
        context::memcpyD2H(host->data(), x->data(), nbytes);
    } else {
        throw std::runtime_error(
            "rearrange_metax: non-contiguous source rearrange not implemented");
    }
    if (y->is_contiguous()) {
        context::memcpyH2D(y->data(), host->data(), nbytes, false);
    } else {
        throw std::runtime_error(
            "rearrange_metax: non-contiguous destination rearrange not implemented");
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    Rearrange::plan_dispatcher().registerDevice(Device::Type::kMetax, &plan);
    Rearrange::run_dispatcher().registerDevice(Device::Type::kMetax, &run);
    Rearrange::cleanup_dispatcher().registerDevice(Device::Type::kMetax, &cleanup);
    return true;
}();

} // namespace
} // namespace infinicore::op::rearrange_impl::metax
