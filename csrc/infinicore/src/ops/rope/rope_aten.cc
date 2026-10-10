#include "infinicore/ops/rope.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/nn/rope.hpp"

namespace infinicore::op::rope_impl::aten {
namespace {

struct PlannedMeta {
    Tensor x_out;
    Tensor x;
    Tensor pos;
    Tensor sin;
    Tensor cos;
    infinicore::nn::RoPE::Algo algo;
};

void *plan(Tensor x_out,
           const Tensor &x,
           const Tensor &pos,
           const Tensor &sin,
           const Tensor &cos,
           infinicore::nn::RoPE::Algo algo) {
    INFINICORE_ASSERT(x_out->device().type() == Device::Type::kQy);
    return new PlannedMeta{x_out, x, pos, sin, cos, algo};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    using namespace ::infinicore::adaptor;
    set_aten_stream_to_infinicore();

    auto x = to_aten_tensor(p->x);
    auto pos = to_aten_tensor(p->pos).to(at::kLong).reshape({-1});
    auto sin_table = to_aten_tensor(p->sin);
    auto cos_table = to_aten_tensor(p->cos);
    auto out = to_aten_tensor(p->x_out);

    const auto rotary_dim = x.size(-1);
    auto sin = sin_table.index_select(/*dim=*/0, pos);
    auto cos = cos_table.index_select(/*dim=*/0, pos);

    // Broadcast sin/cos onto [..., seq, heads, dim] (MiniCPM5 static) or [seq, dim].
    std::vector<int64_t> view_shape(static_cast<size_t>(x.dim()), 1);
    if (x.dim() == 2) {
        view_shape[0] = sin.size(0);
        view_shape[1] = sin.size(1);
    } else {
        INFINICORE_ASSERT(x.dim() >= 3);
        view_shape[static_cast<size_t>(x.dim() - 3)] = sin.size(0);
        view_shape[static_cast<size_t>(x.dim() - 1)] = sin.size(1);
    }
    sin = sin.view(view_shape);
    cos = cos.view(view_shape);

    at::Tensor rotated;
    if (p->algo == infinicore::nn::RoPE::Algo::GPT_NEOX) {
        auto half = rotary_dim / 2;
        auto x1 = x.slice(-1, 0, half);
        auto x2 = x.slice(-1, half, rotary_dim);
        rotated = at::cat({x1 * cos - x2 * sin, x2 * cos + x1 * sin}, /*dim=*/-1);
    } else if (p->algo == infinicore::nn::RoPE::Algo::GPT_J) {
        auto x_even = x.slice(-1, 0, rotary_dim, 2);
        auto x_odd = x.slice(-1, 1, rotary_dim, 2);
        auto out_even = x_even * cos - x_odd * sin;
        auto out_odd = x_odd * cos + x_even * sin;
        rotated = at::empty_like(x);
        rotated.slice(-1, 0, rotary_dim, 2).copy_(out_even);
        rotated.slice(-1, 1, rotary_dim, 2).copy_(out_odd);
    } else {
        throw std::runtime_error("Unsupported RoPE algorithm for QY ATen path");
    }

    out.copy_(rotated);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    RoPE::plan_dispatcher().registerDevice(Device::Type::kQy, &plan, true);
    RoPE::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    RoPE::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::rope_impl::aten
#endif
