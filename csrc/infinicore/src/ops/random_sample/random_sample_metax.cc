#include "infinicore/ops/random_sample.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace infinicore::op::random_sample_impl::metax {
namespace {

static inline float fp16_to_float(uint16_t h) {
	const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
	const uint32_t exp = (h >> 10) & 0x1fu;
	const uint32_t mant = h & 0x3ffu;
	uint32_t out;
	if (exp == 0) {
		if (mant == 0) {
			out = sign;
		} else {
			uint32_t m = mant;
			uint32_t e = 127 - 15 + 1;
			while ((m & 0x400u) == 0) {
				m <<= 1;
				--e;
			}
			m &= 0x3ffu;
			out = sign | (e << 23) | (m << 13);
		}
	} else if (exp == 31) {
		out = sign | 0x7f800000u | (mant << 13);
	} else {
		out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
	}
	float f;
	std::memcpy(&f, &out, sizeof(f));
	return f;
}

static inline float bf16_to_float(uint16_t h) {
	uint32_t out = static_cast<uint32_t>(h) << 16;
	float f;
	std::memcpy(&f, &out, sizeof(f));
	return f;
}

static std::vector<float> logits_to_host_f32(const Tensor &logits) {
	const auto n = logits->numel();
	const auto dtype = logits->dtype();
	std::vector<float> out(n);
	if (dtype == DataType::kFloat32) {
		context::memcpyD2H(out.data(), logits->data(), n * sizeof(float));
		return out;
	}
	if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16) {
		std::vector<uint16_t> raw(n);
		context::memcpyD2H(raw.data(), logits->data(), n * sizeof(uint16_t));
		if (dtype == DataType::kFloat16) {
			for (size_t i = 0; i < n; ++i) {
				out[i] = fp16_to_float(raw[i]);
			}
		} else {
			for (size_t i = 0; i < n; ++i) {
				out[i] = bf16_to_float(raw[i]);
			}
		}
		return out;
	}
	throw std::runtime_error("random_sample_metax: unsupported logits dtype");
}

static int32_t sample_index(std::vector<float> data,
                            float random_val,
                            float topp,
                            int topk,
                            float temperature) {
	const int n = static_cast<int>(data.size());
	if (n <= 0) {
		throw std::runtime_error("random_sample_metax: empty logits");
	}
	if (random_val == 0.0f || topp == 0.0f || topk == 1 || temperature == 0.0f) {
		return static_cast<int32_t>(
		    std::distance(data.begin(), std::max_element(data.begin(), data.end())));
	}
	if (!(temperature > 0.0f)) {
		throw std::runtime_error("random_sample_metax: temperature must be > 0");
	}

	std::vector<int> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](int a, int b) {
		return data[a] > data[b];
	});

	const int k = std::max(1, std::min(topk, n));
	const float maxv = data[order[0]];
	std::vector<float> probs(k);
	float sum = 0.0f;
	for (int i = 0; i < k; ++i) {
		probs[i] = std::exp((data[order[i]] - maxv) / temperature);
		sum += probs[i];
	}
	if (!(sum > 0.0f) || !std::isfinite(sum)) {
		return order[0];
	}
	for (int i = 0; i < k; ++i) {
		probs[i] /= sum;
	}
	float cum = 0.0f;
	float cutoff = topp;
	if (!(cutoff > 0.0f) || cutoff > 1.0f) {
		cutoff = 1.0f;
	}
	std::vector<float> cum_probs(k);
	for (int i = 0; i < k; ++i) {
		cum += probs[i];
		cum_probs[i] = cum;
	}
	const float bound = std::min(cum_probs[k - 1], cutoff) * random_val;
	int idx = 0;
	while (idx < k && cum_probs[idx] < bound) {
		++idx;
	}
	if (idx >= k) {
		idx = k - 1;
	}
	return order[idx];
}

void calculate(Tensor indices,
               Tensor logits,
               float random_val,
               float topp,
               int topk,
               float temperature) {
	INFINICORE_ASSERT(logits->device().type() == Device::Type::kMetax);
	INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, logits);
	INFINICORE_ASSERT(logits->ndim() == 1 && logits->numel() > 0 && logits->is_contiguous());
	INFINICORE_ASSERT(indices->numel() == 1 && indices->is_contiguous());
	INFINICORE_ASSERT(std::isfinite(random_val));
	INFINICORE_ASSERT(std::isfinite(topp));
	INFINICORE_ASSERT(topk > 0);
	INFINICORE_ASSERT(std::isfinite(temperature));

	context::setDevice(logits->device());
	auto host_logits = logits_to_host_f32(logits);
	const int32_t idx = sample_index(std::move(host_logits), random_val, topp, topk, temperature);

	if (indices->dtype() == DataType::kInt32) {
		context::memcpyH2D(indices->data(), &idx, sizeof(int32_t), false);
		return;
	}
	if (indices->dtype() == DataType::kInt64) {
		const int64_t idx64 = static_cast<int64_t>(idx);
		context::memcpyH2D(indices->data(), &idx64, sizeof(int64_t), false);
		return;
	}
	throw std::runtime_error("random_sample_metax: unsupported indices dtype");
}

static bool registered = []() {
	// Host D2H/H2D sampler: used for MetaX and Mars when InfiniOps
	// TopKTopPSamplingFromLogits has no device provider.
	RandomSample::dispatcher().registerDevice(Device::Type::kMetax, &calculate);
	RandomSample::dispatcher().registerDevice(Device::Type::kMars, &calculate);
	return true;
}();

} // namespace
} // namespace infinicore::op::random_sample_impl::metax
