#include "infinicore/ops/causal_conv1d.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include "ops/causal_conv1d/causal_conv1d_metax_launch.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op::causal_conv1d_impl::metax {
namespace {

struct PlannedMeta {
	Tensor out;
	Tensor conv_state;
	std::optional<Tensor> final_conv_state;
	Tensor qkv;
	Tensor weight;
	std::optional<Tensor> bias;
	std::optional<Tensor> cu_seqlens;
	std::optional<Tensor> initial_state_indices;
	std::optional<Tensor> final_state_indices;
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
		throw std::runtime_error("causal_conv1d_metax: unsupported dtype");
	}
}

static bool is_i64(DataType dtype) {
	return dtype == DataType::kInt64;
}

void *plan(Tensor out,
           Tensor conv_state,
           std::optional<Tensor> final_conv_state,
           const Tensor &qkv,
           const Tensor &weight,
           std::optional<Tensor> bias,
           std::optional<Tensor> cu_seqlens,
           std::optional<Tensor> initial_state_indices,
           std::optional<Tensor> final_state_indices) {
	INFINICORE_ASSERT(out->device().type() == Device::Type::kMetax);
	INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, conv_state, qkv, weight);
	if (final_conv_state.has_value()) {
		INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_conv_state.value());
	}
	if (bias.has_value()) {
		INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, bias.value());
	}
	if (cu_seqlens.has_value()) {
		INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, cu_seqlens.value());
	}
	if (initial_state_indices.has_value()) {
		INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, initial_state_indices.value());
	}
	if (final_state_indices.has_value()) {
		INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, final_state_indices.value());
	}
	if (weight->shape().size() != 3 || weight->shape()[2] != 4) {
		throw std::runtime_error("causal_conv1d_metax: only kernel_size==4 is supported");
	}
	return new PlannedMeta{out,
	                       conv_state,
	                       final_conv_state,
	                       qkv,
	                       weight,
	                       bias,
	                       cu_seqlens,
	                       initial_state_indices,
	                       final_state_indices};
}

void run(void *planned_meta) {
	auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
	context::setDevice(p->out->device());

	const auto &qkv_shape = p->qkv->shape();
	const auto &state_shape = p->conv_state->shape();
	const auto &weight_shape = p->weight->shape();

	const bool has_cu = p->cu_seqlens.has_value();
	const bool indexed = p->initial_state_indices.has_value() || p->final_state_indices.has_value();

	size_t request_count = 0;
	size_t T = 0;
	size_t total_tokens = 0;
	if (has_cu) {
		const auto &cu_shape = p->cu_seqlens.value()->shape();
		if (cu_shape.size() != 1 || cu_shape[0] < 2) {
			throw std::runtime_error("causal_conv1d_metax: bad cu_seqlens shape");
		}
		request_count = cu_shape[0] - 1;
		total_tokens = qkv_shape[1];
		T = total_tokens; // unused when has_cu; kernel reads cu_seqlens
	} else {
		request_count = qkv_shape[0];
		T = qkv_shape[1];
		total_tokens = T;
	}

	const size_t C = weight_shape[0];
	const size_t pool_size = state_shape[0];

	InfiniLMCausalConv1dLaunchArgs args{};
	args.dtype = dtype_code(p->qkv->dtype());
	args.request_count = static_cast<int>(request_count);
	args.C = static_cast<int>(C);
	args.T = static_cast<int>(T);
	args.total_tokens = static_cast<int>(total_tokens);
	args.pool_size = static_cast<int>(pool_size);
	args.has_bias = p->bias.has_value() ? 1 : 0;
	args.has_cu_seqlens = has_cu ? 1 : 0;
	args.cu_seqlens_i64 = has_cu && is_i64(p->cu_seqlens.value()->dtype()) ? 1 : 0;
	args.initial_state_indices_i64 =
	    p->initial_state_indices.has_value() && is_i64(p->initial_state_indices.value()->dtype()) ? 1 : 0;
	args.final_state_indices_i64 =
	    p->final_state_indices.has_value() && is_i64(p->final_state_indices.value()->dtype()) ? 1 : 0;
	args.indexed_state_pool = indexed ? 1 : 0;

	const auto &os = p->out->strides();
	const auto &ss = p->conv_state->strides();
	const auto &qs = p->qkv->strides();
	const auto &ws = p->weight->strides();
	args.out_s0 = os[0];
	args.out_s1 = os[1];
	args.out_s2 = os[2];
	args.state_s0 = ss[0];
	args.state_s1 = ss[1];
	args.state_s2 = ss[2];
	if (p->final_conv_state.has_value()) {
		const auto &fs = p->final_conv_state.value()->strides();
		args.final_s0 = fs[0];
		args.final_s1 = fs[1];
		args.final_s2 = fs[2];
	} else {
		args.final_s0 = args.final_s1 = args.final_s2 = 0;
	}
	args.qkv_s0 = qs[0];
	args.qkv_s1 = qs[1];
	args.qkv_s2 = qs[2];
	args.weight_s0 = ws[0];
	args.weight_s1 = ws[1];
	args.weight_s2 = ws[2];
	args.bias_s0 = p->bias.has_value() ? p->bias.value()->strides()[0] : 0;

	args.out = p->out->data();
	args.conv_state = p->conv_state->data();
	args.final_conv_state =
	    p->final_conv_state.has_value() ? p->final_conv_state.value()->data() : nullptr;
	args.qkv = p->qkv->data();
	args.weight = p->weight->data();
	args.bias = p->bias.has_value() ? p->bias.value()->data() : nullptr;
	args.cu_seqlens = has_cu ? p->cu_seqlens.value()->data() : nullptr;
	args.initial_state_indices =
	    p->initial_state_indices.has_value() ? p->initial_state_indices.value()->data() : nullptr;
	args.final_state_indices =
	    p->final_state_indices.has_value() ? p->final_state_indices.value()->data() : nullptr;
	args.stream = reinterpret_cast<void *>(context::getStream());

	const int rc = infinilm_causal_conv1d_metax_launch(&args);
	if (rc != 0) {
		throw std::runtime_error("causal_conv1d_metax: kernel launch failed, code=" +
		                         std::to_string(rc));
	}
}

void cleanup(void **planned_meta_ptr) {
	delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
	*planned_meta_ptr = nullptr;
}

static bool registered = []() {
	CausalConv1d::plan_dispatcher().registerDevice(Device::Type::kMetax, &plan);
	CausalConv1d::run_dispatcher().registerDevice(Device::Type::kMetax, &run);
	CausalConv1d::cleanup_dispatcher().registerDevice(Device::Type::kMetax, &cleanup);
	return true;
}();

} // namespace
} // namespace infinicore::op::causal_conv1d_impl::metax
