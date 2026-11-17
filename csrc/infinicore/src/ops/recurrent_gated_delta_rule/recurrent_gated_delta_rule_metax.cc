#include "infinicore/ops/recurrent_gated_delta_rule.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "ops/recurrent_gated_delta_rule/recurrent_gated_delta_rule_metax_launch.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op::recurrent_gated_delta_rule_impl::metax {
namespace {

struct PlannedMeta {
	Tensor out;
	Tensor initial_state;
	std::optional<Tensor> final_state;
	Tensor q;
	Tensor k;
	Tensor v;
	Tensor g;
	Tensor beta;
	std::optional<Tensor> initial_state_indices;
	std::optional<Tensor> final_state_indices;
	bool use_qk_l2norm;
};

static int dtype_code(DataType dtype) {
	switch (dtype) {
	case DataType::kFloat32: return 0;
	case DataType::kFloat16: return 1;
	case DataType::kBFloat16: return 2;
	default: throw std::runtime_error("recurrent_gated_delta_rule_metax: unsupported dtype");
	}
}

static bool is_i64(DataType dtype) { return dtype == DataType::kInt64; }

void *plan(Tensor out,
           Tensor initial_state,
           std::optional<Tensor> final_state,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &g,
           const Tensor &beta,
           std::optional<Tensor> initial_state_indices,
           std::optional<Tensor> final_state_indices,
           bool use_qk_l2norm) {
	INFINICORE_ASSERT(out->device().type() == Device::Type::kMetax);
	INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state, q, k, v, g, beta);
	return new PlannedMeta{out, initial_state, final_state, q, k, v, g, beta,
	                       initial_state_indices, final_state_indices, use_qk_l2norm};
}

void run(void *planned_meta) {
	auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
	context::setDevice(p->out->device());

	const auto &qshape = p->q->shape();
	const auto &vshape = p->v->shape();
	const auto &istate = p->initial_state->shape();
	const bool indexed = p->initial_state_indices.has_value() || p->final_state_indices.has_value();
	const size_t B = qshape[0];
	const size_t T = qshape[1];
	const size_t Hk = qshape[2];
	const size_t Hv = vshape[2];
	const size_t Dk = qshape[3];
	const size_t Dv = vshape[3];
	if (Hk == 0 || Hv % Hk != 0) {
		throw std::runtime_error("recurrent_gated_delta_rule_metax: invalid head grouping");
	}

	InfiniLMRecurrentGatedLaunchArgs args{};
	args.dtype = dtype_code(p->q->dtype());
	args.gate_dtype = dtype_code(p->g->dtype());
	args.B = static_cast<int>(B);
	args.T = static_cast<int>(T);
	args.Hk = static_cast<int>(Hk);
	args.Hv = static_cast<int>(Hv);
	args.Dk = static_cast<int>(Dk);
	args.Dv = static_cast<int>(Dv);
	args.pool_size = static_cast<int>(istate[0]);
	args.value_heads_per_key_head = static_cast<int>(Hv / Hk);
	args.use_qk_l2norm = p->use_qk_l2norm ? 1 : 0;
	args.indexed_state_pool = indexed ? 1 : 0;
	args.initial_state_indices_i64 =
	    p->initial_state_indices.has_value() && is_i64(p->initial_state_indices.value()->dtype()) ? 1 : 0;
	args.final_state_indices_i64 =
	    p->final_state_indices.has_value() && is_i64(p->final_state_indices.value()->dtype()) ? 1 : 0;
	args.has_final_state = p->final_state.has_value() ? 1 : 0;

	const auto &os = p->out->strides();
	const auto &is = p->initial_state->strides();
	args.out_s0 = os[0]; args.out_s1 = os[1]; args.out_s2 = os[2];
	args.initial_s0 = is[0]; args.initial_s1 = is[1]; args.initial_s2 = is[2]; args.initial_s3 = is[3];
	if (p->final_state.has_value()) {
		const auto &fs = p->final_state.value()->strides();
		args.final_s0 = fs[0]; args.final_s1 = fs[1]; args.final_s2 = fs[2]; args.final_s3 = fs[3];
	}
	const auto &qs = p->q->strides();
	const auto &ks = p->k->strides();
	const auto &vs = p->v->strides();
	const auto &gs = p->g->strides();
	const auto &bs = p->beta->strides();
	args.q_s0 = qs[0]; args.q_s1 = qs[1]; args.q_s2 = qs[2];
	args.k_s0 = ks[0]; args.k_s1 = ks[1]; args.k_s2 = ks[2];
	args.v_s0 = vs[0]; args.v_s1 = vs[1]; args.v_s2 = vs[2];
	args.g_s0 = gs[0]; args.g_s1 = gs[1]; args.g_s2 = gs[2];
	args.beta_s0 = bs[0]; args.beta_s1 = bs[1]; args.beta_s2 = bs[2];

	args.out = p->out->data();
	args.initial_state = p->initial_state->data();
	args.final_state = p->final_state.has_value() ? p->final_state.value()->data() : p->initial_state->data();
	args.q = p->q->data();
	args.k = p->k->data();
	args.v = p->v->data();
	args.g = p->g->data();
	args.beta = p->beta->data();
	args.initial_state_indices =
	    p->initial_state_indices.has_value() ? p->initial_state_indices.value()->data() : nullptr;
	args.final_state_indices =
	    p->final_state_indices.has_value() ? p->final_state_indices.value()->data() : nullptr;
	args.stream = reinterpret_cast<void *>(context::getStream());

	const int rc = infinilm_recurrent_gated_delta_rule_metax_launch(&args);
	if (rc != 0) {
		throw std::runtime_error("recurrent_gated_delta_rule_metax: launch failed code=" + std::to_string(rc));
	}
}

void cleanup(void **planned_meta_ptr) {
	delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
	*planned_meta_ptr = nullptr;
}

static bool registered = []() {
	RecurrentGatedDeltaRule::plan_dispatcher().registerDevice(Device::Type::kMetax, &plan);
	RecurrentGatedDeltaRule::run_dispatcher().registerDevice(Device::Type::kMetax, &run);
	RecurrentGatedDeltaRule::cleanup_dispatcher().registerDevice(Device::Type::kMetax, &cleanup);
	return true;
}();

} // namespace
} // namespace infinicore::op::recurrent_gated_delta_rule_impl::metax
