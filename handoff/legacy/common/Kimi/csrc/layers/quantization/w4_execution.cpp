#include "execution/submit.hpp"
#include "w4_execution.hpp"
#include "w4_marlin_adapter.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "runtime/runtime.hpp"
#include <stdexcept>

namespace infinilm::quantization {
using infinilm::runtime::Tensor;
using infinilm::runtime::Runtime;
namespace { void require(bool ok, const char* msg) { if (!ok) throw std::invalid_argument(msg); } }
W4Execution::W4Execution(Runtime& runtime, const ParamsMap& raw, const ParamsMap& prepared,
                        size_t k, size_t n, int scheme, int group, bool full)
    : owner_(runtime.shared_from_this()), raw_leases_(raw), weight_(prepared.at("qweight")),
      scales_(prepared.at("scales")), zeros_(prepared.at("qzeros")), gidx_(prepared.at("g_idx")),
      perm_(prepared.at("perm")), bias_(prepared.count("bias") ? prepared.at("bias") : Tensor()),
      empty_bias_(prepared.at("empty_bias")), k_(k), n_(n), scheme_(scheme), group_(group), full_(full) {}
std::shared_ptr<const W4Execution> W4Execution::prepare(Runtime& runtime, const ParamsMap& raw,
    const nlohmann::json& config, int split_dim, bool retained_global_scale_table,
    std::optional<int64_t> expected_non_desc_group_offset) {
    // Adapter completes its final stream fence before returning. All map copies,
    // pointer binding and allocation below are still private and may throw.
    auto result = prepare_w4_marlin(runtime, raw, config, split_dim, retained_global_scale_table, expected_non_desc_group_offset);
    return std::shared_ptr<const W4Execution>(new W4Execution(runtime, raw, result.params,
        result.k, result.n, result.scheme, result.group, result.full));
}
Tensor W4Execution::forward(Runtime& runtime, const Tensor& input, bool has_bias, float alpha) const {
#ifdef INFINILM_W4_MARLIN_OPS_UNAVAILABLE
    runtime.assertOwnerThread();
    throw std::runtime_error("W4 Marlin public APIs are unavailable in this SDK; AWQ/GPTQ execution is unsupported");
#else

    runtime.assertOwnerThread();
    auto owner = owner_.lock();
    require(owner && owner.get() == &runtime, "W4 binding belongs to a different or expired Runtime");
    require(input && input->ndim() > 0 && input->shape().back() == k_ &&
            input->device() == runtime.device() && input->dtype() == scales_->dtype(), "W4 input differs from fixed binding");
    require(!has_bias || bool(bias_), "W4 binding has no bias");
    Tensor a, output;
    try {
        a = w4_detail::contiguous(runtime, input);
        auto shape = a->shape(); shape.back() = n_;
        output = Tensor::empty(runtime, shape, a->dtype());
        const size_t m = a->numel() / k_;
        const auto out_view = output->view({m,n_});
        const auto a_view = a->view({m,k_});
        infinilm::execution::submit<infini::ops::W4MarlinGemm>(
            runtime, infinilm::execution::RecordingPolicy::HostOnly,
            infini::ops::Config::Automatic(), a_view, weight_, scales_, zeros_,
            gidx_, perm_, has_bias ? bias_ : empty_bias_,
            static_cast<int64_t>(scheme_), static_cast<int64_t>(group_), full_, alpha, out_view);
        return output;
    } catch (...) { w4_detail::drain_or_terminate(runtime); throw; }

#endif
}
Tensor W4Execution::forward_allreduce(Runtime& runtime, const Tensor& input, bool has_bias,
    infinicclComm_t communicator, float alpha) const {
    runtime.assertOwnerThread();
    auto owner = owner_.lock();
    require(owner && owner.get() == &runtime, "W4 binding belongs to a different or expired Runtime");
    require(!has_bias || bool(bias_), "W4 binding has no bias");
    Tensor output;
    try {
        output = forward(runtime, input, false, alpha);
        infinilm::engine::distributed::allreduce_(runtime, output, output, infinicclSum, communicator);
        if (has_bias) {
            infinilm::execution::submit<infini::ops::Add>(
                runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), output, bias_, output);
            INFINILM_TRACE_OP(Add, output);
        }
        return output;
    } catch (...) { w4_detail::drain_or_terminate(runtime); throw; }
}
} // namespace infinilm::quantization
