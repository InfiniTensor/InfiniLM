#include "infinicore/ops/mul_scalar.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

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
    using namespace ::infinicore::adaptor;
    set_aten_stream_to_infinicore();
    to_aten_tensor(p->c).copy_(to_aten_tensor(p->a) * p->alpha);
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
