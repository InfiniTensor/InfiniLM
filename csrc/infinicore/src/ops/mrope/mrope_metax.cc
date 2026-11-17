#include "infinicore/ops/mrope.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "ops/mrope/mrope_metax_launch.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op::mrope_impl::metax {
namespace {

struct PlannedMeta {
	Tensor q_out;
	Tensor k_out;
	Tensor q;
	Tensor k;
	Tensor cos;
	Tensor sin;
	Tensor positions;
	int head_size;
	int rotary_dim;
	int section_t;
	int section_h;
	int section_w;
	bool interleaved;
};

static int dtype_code(DataType dtype) {
	switch (dtype) {
	case DataType::kFloat32: return 0;
	case DataType::kFloat16: return 1;
	case DataType::kBFloat16: return 2;
	default: throw std::runtime_error("mrope_metax: unsupported dtype");
	}
}

void *plan(Tensor q_out,
           Tensor k_out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &cos,
           const Tensor &sin,
           const Tensor &positions,
           int head_size,
           int rotary_dim,
           int section_t,
           int section_h,
           int section_w,
           bool interleaved) {
	INFINICORE_ASSERT(q->device().type() == Device::Type::kMetax);
	INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_out, k_out, q, k, cos, sin, positions);
	return new PlannedMeta{q_out, k_out, q, k, cos, sin, positions,
	                       head_size, rotary_dim, section_t, section_h, section_w, interleaved};
}

void run(void *planned_meta) {
	auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
	context::setDevice(p->q->device());

	const auto &qshape = p->q->shape();
	const auto &kshape = p->k->shape();
	if (qshape.size() != 2 || kshape.size() != 2) {
		throw std::runtime_error("mrope_metax: expected q/k shape [tokens, heads*head_size]");
	}
	if (qshape[1] % static_cast<size_t>(p->head_size) != 0 ||
	    kshape[1] % static_cast<size_t>(p->head_size) != 0) {
		throw std::runtime_error("mrope_metax: last dim not divisible by head_size");
	}
	const size_t num_tokens = qshape[0];
	const size_t num_q_heads = qshape[1] / static_cast<size_t>(p->head_size);
	const size_t num_kv_heads = kshape[1] / static_cast<size_t>(p->head_size);

	const auto &pos_shape = p->positions->shape();
	const bool positions_has_axes = pos_shape.size() == 2 && pos_shape[0] == 3;

	InfiniLMMRoPELaunchArgs args{};
	args.dtype = dtype_code(p->q->dtype());
	args.pos_i64 = (p->positions->dtype() == DataType::kInt64) ? 1 : 0;
	args.num_tokens = static_cast<int>(num_tokens);
	args.num_q_heads = static_cast<int>(num_q_heads);
	args.num_kv_heads = static_cast<int>(num_kv_heads);
	args.head_size = p->head_size;
	args.rotary_dim = p->rotary_dim;
	args.section_t = p->section_t;
	args.section_h = p->section_h;
	args.section_w = p->section_w;
	args.interleaved = p->interleaved ? 1 : 0;
	args.positions_has_axes = positions_has_axes ? 1 : 0;
	args.max_position_embeddings = static_cast<int>(p->cos->shape()[0]);

	// Infiniop convention: 2D [tokens, H*D] with contiguous last dim; head stride = head_size.
	args.q_out_s_token = p->q_out->strides()[0];
	args.q_out_s_head = p->head_size;
	args.k_out_s_token = p->k_out->strides()[0];
	args.k_out_s_head = p->head_size;
	args.q_s_token = p->q->strides()[0];
	args.q_s_head = p->head_size;
	args.k_s_token = p->k->strides()[0];
	args.k_s_head = p->head_size;

	const auto &cs = p->cos->strides();
	const auto &ss = p->sin->strides();
	args.cos_s_pos = cs[0];
	args.sin_s_pos = ss[0];
	const auto &ps = p->positions->strides();
	if (positions_has_axes) {
		args.pos_s_axis = ps[0];
		args.pos_s_token = ps[1];
	} else {
		args.pos_s_axis = 0;
		args.pos_s_token = ps[0];
	}

	args.q_out = p->q_out->data();
	args.k_out = p->k_out->data();
	args.q = p->q->data();
	args.k = p->k->data();
	args.cos = p->cos->data();
	args.sin = p->sin->data();
	args.positions = p->positions->data();
	args.stream = reinterpret_cast<void *>(context::getStream());

	const int rc = infinilm_mrope_metax_launch(&args);
	if (rc != 0) {
		throw std::runtime_error("mrope_metax: launch failed code=" + std::to_string(rc));
	}
}

void cleanup(void **planned_meta_ptr) {
	delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
	*planned_meta_ptr = nullptr;
}

static bool registered = []() {
	MRoPE::plan_dispatcher().registerDevice(Device::Type::kMetax, &plan);
	MRoPE::run_dispatcher().registerDevice(Device::Type::kMetax, &run);
	MRoPE::cleanup_dispatcher().registerDevice(Device::Type::kMetax, &cleanup);
	return true;
}();

} // namespace
} // namespace infinicore::op::mrope_impl::metax
