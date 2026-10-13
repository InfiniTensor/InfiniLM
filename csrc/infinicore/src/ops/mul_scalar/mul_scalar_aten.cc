#include "infinicore/ops/mul_scalar.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/context/context.hpp"

#include <mul_scalar_kernel.h>

namespace infinicore::op::mul_scalar_impl::aten {
namespace {

struct PlannedMeta {
    Tensor c;
    Tensor a;
    double alpha;
};

void *plan(Tensor c, const Tensor &a, double alpha) {
    INFINICORE_ASSERT(c->device().type() == Device::Type::kQy);
    return new PlannedMeta{c, a, alpha};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_ASSERT(p->c->device() == p->a->device());
    INFINICORE_ASSERT(p->c->dtype() == p->a->dtype());
    INFINICORE_ASSERT(p->c->numel() == p->a->numel());
    INFINICORE_ASSERT(p->c->is_contiguous());

    Tensor a = p->a->is_contiguous() ? p->a : p->a->contiguous();
    INFINICORE_ASSERT(a->is_contiguous());
    INFINICORE_ASSERT(a->numel() == p->c->numel());
    INFINICORE_ASSERT(a->dtype() == p->c->dtype());

    ::infini::ops::qy::mul_scalar_launch(
        p->c->data(), a->data(), static_cast<size_t>(a->numel()),
        static_cast<int>(a->dtype()), p->alpha,
        reinterpret_cast<void *>(context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    MulScalar::plan_dispatcher().registerDevice(Device::Type::kQy, &plan, true);
    MulScalar::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MulScalar::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::mul_scalar_impl::aten
#endif
