#include "infinicore/ops/fused_gated_delta_net_gating.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "ops/fused_gated_delta_net_gating/fused_gated_delta_net_gating_metax_launch.h"

#include <stdexcept>
#include <string>

namespace infinicore::op::fused_gated_delta_net_gating_impl::metax {
namespace {

struct PlannedMeta {
	Tensor g;
	Tensor beta_output;
	Tensor A_log;
	Tensor a;
	Tensor b;
	Tensor dt_bias;
	float beta;
	float threshold;
};

static int dtype_code(DataType dtype) {
	switch (dtype) {
	case DataType::kFloat32:
		return 0;
	case DataType::kFloat16:
		return 1;
	case DataType::kBFloat16:
		return 2;
	default:
		throw std::runtime_error("fused_gated_delta_net_gating_metax: unsupported dtype");
	}
}

void *plan(Tensor g,
           Tensor beta_output,
           const Tensor &A_log,
           const Tensor &a,
           const Tensor &b,
           const Tensor &dt_bias,
           float beta,
           float threshold) {
	INFINICORE_ASSERT(g->device().type() == Device::Type::kMetax);
	INFINICORE_ASSERT_TENSORS_SAME_DEVICE(g, beta_output, A_log, a, b, dt_bias);
	if (g->dtype() != DataType::kFloat32 || beta_output->dtype() != DataType::kFloat32) {
		throw std::runtime_error("fused_gated_delta_net_gating_metax: g/beta_output must be f32");
	}
	return new PlannedMeta{g, beta_output, A_log, a, b, dt_bias, beta, threshold};
}

void run(void *planned_meta) {
	auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
	context::setDevice(p->g->device());

	const auto &ashape = p->a->shape();
	if (ashape.size() != 3) {
		throw std::runtime_error("fused_gated_delta_net_gating_metax: a must be [B,S,H]");
	}

	InfiniLMFusedGatingLaunchArgs args{};
	args.dtype = dtype_code(p->a->dtype());
	args.batch = static_cast<int>(ashape[0]);
	args.seq_len = static_cast<int>(ashape[1]);
	args.hidden = static_cast<int>(ashape[2]);
	args.beta = p->beta;
	args.threshold = p->threshold;

	const auto &gs = p->g->strides();
	const auto &bs = p->beta_output->strides();
	const auto &As = p->A_log->strides();
	const auto &as = p->a->strides();
	const auto &bbs = p->b->strides();
	const auto &ds = p->dt_bias->strides();
	args.g_s0 = gs[0];
	args.g_s1 = gs[1];
	args.g_s2 = gs[2];
	args.beta_s0 = bs[0];
	args.beta_s1 = bs[1];
	args.beta_s2 = bs[2];
	args.A_log_s0 = As[0];
	args.a_s0 = as[0];
	args.a_s1 = as[1];
	args.a_s2 = as[2];
	args.b_s0 = bbs[0];
	args.b_s1 = bbs[1];
	args.b_s2 = bbs[2];
	args.dt_bias_s0 = ds[0];

	args.g = reinterpret_cast<float *>(p->g->data());
	args.beta_output = reinterpret_cast<float *>(p->beta_output->data());
	args.A_log = p->A_log->data();
	args.a = p->a->data();
	args.b = p->b->data();
	args.dt_bias = p->dt_bias->data();
	args.stream = reinterpret_cast<void *>(context::getStream());

	const int rc = infinilm_fused_gated_delta_net_gating_metax_launch(&args);
	if (rc != 0) {
		throw std::runtime_error("fused_gated_delta_net_gating_metax: launch failed code=" +
		                         std::to_string(rc));
	}
}

void cleanup(void **planned_meta_ptr) {
	delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
	*planned_meta_ptr = nullptr;
}

static bool registered = []() {
	FusedGatedDeltaNetGating::plan_dispatcher().registerDevice(Device::Type::kMetax, &plan);
	FusedGatedDeltaNetGating::run_dispatcher().registerDevice(Device::Type::kMetax, &run);
	FusedGatedDeltaNetGating::cleanup_dispatcher().registerDevice(Device::Type::kMetax, &cleanup);
	return true;
}();

} // namespace
} // namespace infinicore::op::fused_gated_delta_net_gating_impl::metax
