#include "infinicore/ops/chunk_gated_delta_rule.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "ops/chunk_gated_delta_rule/chunk_gated_delta_rule_metax_launch.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op::chunk_gated_delta_rule_impl::metax {
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
	std::optional<Tensor> cu_seqlens;
	std::optional<Tensor> initial_state_indices;
	std::optional<Tensor> final_state_indices;
	bool use_qk_l2norm;
	size_t chunk_size;
	Tensor workspace;
};

static int dtype_code(DataType dtype) {
	switch (dtype) {
	case DataType::kFloat32: return 0;
	case DataType::kFloat16: return 1;
	case DataType::kBFloat16: return 2;
	default: throw std::runtime_error("chunk_gated_delta_rule_metax: unsupported dtype");
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
           std::optional<Tensor> cu_seqlens,
           std::optional<Tensor> initial_state_indices,
           std::optional<Tensor> final_state_indices,
           bool use_qk_l2norm,
           size_t chunk_size) {
	INFINICORE_ASSERT(out->device().type() == Device::Type::kMetax);
	INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state, q, k, v, g, beta);
	const auto &vshape = v->shape();
	const size_t B = cu_seqlens.has_value() ? (cu_seqlens.value()->shape()[0] - 1) : vshape[0];
	const size_t Hv = vshape[2];
	const size_t Dk = q->shape()[3];
	const size_t Dv = vshape[3];
	auto workspace = Tensor::empty({B * Hv * Dk * Dv}, DataType::kFloat32, out->device());
	return new PlannedMeta{out, initial_state, final_state, q, k, v, g, beta,
	                       cu_seqlens, initial_state_indices, final_state_indices,
	                       use_qk_l2norm, chunk_size, workspace};
}

void run(void *planned_meta) {
	auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
	context::setDevice(p->out->device());

	const auto &qshape = p->q->shape();
	const auto &vshape = p->v->shape();
	const auto &istate = p->initial_state->shape();

	const bool has_cu = p->cu_seqlens.has_value();
	const bool indexed = p->initial_state_indices.has_value() || p->final_state_indices.has_value();
	const size_t B = has_cu ? (p->cu_seqlens.value()->shape()[0] - 1) : vshape[0];
	const size_t T = qshape[1];
	const size_t Hk = qshape[2];
	const size_t Hv = vshape[2];
	const size_t Dk = qshape[3];
	const size_t Dv = vshape[3];
	if (Hk == 0 || Hv % Hk != 0) {
		throw std::runtime_error("chunk_gated_delta_rule_metax: invalid head grouping");
	}

	InfiniLMChunkGatedLaunchArgs args{};
	args.dtype = dtype_code(p->q->dtype());
	args.gate_dtype = dtype_code(p->g->dtype());
	args.B = static_cast<int>(B);
	args.T = static_cast<int>(T);
	args.Hk = static_cast<int>(Hk);
	args.Hv = static_cast<int>(Hv);
	args.Dk = static_cast<int>(Dk);
	args.Dv = static_cast<int>(Dv);
	args.chunk_size = static_cast<int>(p->chunk_size);
	args.pool_size = static_cast<int>(istate[0]);
	args.value_heads_per_key_head = static_cast<int>(Hv / Hk);
	args.use_qk_l2norm = p->use_qk_l2norm ? 1 : 0;
	args.has_cu_seqlens = has_cu ? 1 : 0;
	args.indexed_state_pool = indexed ? 1 : 0;
	args.cu_seqlens_i64 = has_cu && is_i64(p->cu_seqlens.value()->dtype()) ? 1 : 0;
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

	args.workspace = p->workspace->data();
	args.out = p->out->data();
	args.initial_state = p->initial_state->data();
	args.final_state = p->final_state.has_value() ? p->final_state.value()->data() : p->initial_state->data();
	args.q = p->q->data();
	args.k = p->k->data();
	args.v = p->v->data();
	args.g = p->g->data();
	args.beta = p->beta->data();
	args.cu_seqlens = has_cu ? p->cu_seqlens.value()->data() : nullptr;
	args.initial_state_indices =
	    p->initial_state_indices.has_value() ? p->initial_state_indices.value()->data() : nullptr;
	args.final_state_indices =
	    p->final_state_indices.has_value() ? p->final_state_indices.value()->data() : nullptr;
	args.stream = reinterpret_cast<void *>(context::getStream());

	const int rc = infinilm_chunk_gated_delta_rule_metax_launch(&args);
	if (rc != 0) {
		throw std::runtime_error("chunk_gated_delta_rule_metax: launch failed code=" + std::to_string(rc));
	}
}

void cleanup(void **planned_meta_ptr) {
	delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
	*planned_meta_ptr = nullptr;
}

static bool registered = []() {
	ChunkGatedDeltaRule::plan_dispatcher().registerDevice(Device::Type::kMetax, &plan);
	ChunkGatedDeltaRule::run_dispatcher().registerDevice(Device::Type::kMetax, &run);
	ChunkGatedDeltaRule::cleanup_dispatcher().registerDevice(Device::Type::kMetax, &cleanup);
	return true;
}();

} // namespace
} // namespace infinicore::op::chunk_gated_delta_rule_impl::metax
