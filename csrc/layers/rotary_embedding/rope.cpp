#include "rope.hpp"
#include "../../execution/submit.hpp"
#include "runtime/custom_types.h"
#include "runtime/utils.hpp"
#include "runtime/runtime.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::layers {
using infinilm::runtime::Tensor;
using infinilm::runtime::Runtime;
using infinilm::runtime::DataType;
using infinilm::runtime::Device;
using infinilm::runtime::Shape;
using infinilm::runtime::Size;

namespace {

template <typename T>
std::vector<T> combine_cos_sin_cache(const std::vector<T> &cos,
                                     const std::vector<T> &sin,
                                     size_t max_seq_len,
                                     size_t cache_dim) {
    std::vector<T> combined(max_seq_len * cache_dim * 2);
    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        std::copy_n(cos.begin() + pos * cache_dim, cache_dim,
                    combined.begin() + pos * cache_dim * 2);
        std::copy_n(sin.begin() + pos * cache_dim, cache_dim,
                    combined.begin() + (pos * 2 + 1) * cache_dim);
    }
    return combined;
}

// Called while every dtype/combined source vector is still alive in initialize_cache.
void upload_cache(Runtime &runtime, Tensor table, const Tensor &host) {
    try {
        table->copy_from(runtime, host);
    } catch (...) {
        const auto failure = std::current_exception();
        if (runtime.device().type() != Device::Type::kCpu) {
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
        }
        std::rethrow_exception(failure);
    }
}

} // namespace

RoPE::RoPE(Runtime &runtime, size_t head_dim,
           size_t rotary_dim,
           size_t max_seq_len,
           double theta,
           Algo algo,
           const DataType &dtype,
           const Device &device,
           std::shared_ptr<RopeScalingConfig> scaling,
           std::optional<std::vector<int>> mrope_section,
           bool mrope_interleaved)
    : rotary_dim_(rotary_dim),
      head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      theta_(theta),
      algo_(algo),
      dtype_(dtype),
      scaling_(scaling),
      mrope_section_(mrope_section),
      mrope_interleaved_(mrope_interleaved) {
    runtime.assertOwnerThread();
    if (runtime.device() != device) {
        throw std::invalid_argument("RoPE cache construction Runtime and device must agree");
    }
    if (rotary_dim % 2 != 0) {
        throw std::invalid_argument("rotary_dim must be even for RoPE, got " + std::to_string(rotary_dim));
    }
    if (rotary_dim == 0 || rotary_dim > head_dim_)
        throw std::invalid_argument("RoPE rotary_dim must be positive and no larger than head_dim");
    if (algo != Algo::GPT_J && algo != Algo::GPT_NEOX)
        throw std::invalid_argument("RoPE algorithm must be GPT_J or GPT_NEOX");
    if (mrope_section_.has_value()) {
        if (head_dim_ > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("MRoPE head_dim exceeds the public int ABI");
        const auto &section = mrope_section_.value();
        if (section.size() != 3 || section[0] <= 0 || section[1] <= 0 || section[2] <= 0) {
            throw std::invalid_argument("mrope_section must contain 3 positive values");
        }
        if (2 * (int64_t{section[0]} + section[1] + section[2]) != static_cast<int64_t>(rotary_dim_)) {
            throw std::invalid_argument("MRoPE section sum must equal rotary_dim / 2");
        }
    }
    device_ = device;

    // Initialize cache tables
    cache_runtime_owner_ = runtime.shared_from_this();
    initialize_cache(runtime);
}

void RoPE::initialize_cache(Runtime &runtime) {
    size_t cache_dim = rotary_dim_ / 2;

    Tensor sin_cache_next;
    Tensor cos_cache_next;
    Tensor cos_sin_cache_next;

    if ((device_.type() == Device::Type::kNvidia
         || device_.type() == Device::Type::kMetax
         || device_.type() == Device::Type::kIluvatar
         || device_.type() == Device::Type::kMoore
         || device_.type() == Device::Type::kCambricon
         || device_.type() == Device::Type::kAscend
         || device_.type() == Device::Type::kHygon)
        && !mrope_section_) {
        cos_sin_cache_next = Tensor::empty_parameter(runtime, {max_seq_len_, rotary_dim_}, dtype_);
    }

    if (!cos_sin_cache_next) {
        sin_cache_next = Tensor::empty_parameter(runtime, {max_seq_len_, cache_dim}, dtype_);
        cos_cache_next = Tensor::empty_parameter(runtime, {max_seq_len_, cache_dim}, dtype_);
    }

    // Pre-compute sin and cos values
    // Frequency generation always uses GPT-J style (theta^(-2j/rotary_dim)).
    // The rotation algorithm (algo_) controls how dimensions are paired in the kernel.

    // Compute on CPU first, then copy to device
    auto cpu_device = Device(Device::Type::kCpu, 0);

    // Allocate CPU buffers
    std::vector<float> sin_data(max_seq_len_ * cache_dim);
    std::vector<float> cos_data(max_seq_len_ * cache_dim);

    // Base frequencies depend only on the dimension, not the position. Keep
    // the exact existing FP32 expression; scaling can still depend on position.
    std::vector<float> base_inv_frequencies(cache_dim);
    for (size_t dim_idx = 0; dim_idx < cache_dim; ++dim_idx) {
        base_inv_frequencies[dim_idx] = 1.0f / std::pow(
            static_cast<float>(theta_),
            2.0f * static_cast<float>(dim_idx) / static_cast<float>(rotary_dim_));
    }

    for (size_t pos = 0; pos < max_seq_len_; pos++) {
        for (size_t dim_idx = 0; dim_idx < cache_dim; dim_idx++) {
            // 1. Base inverse frequency (shared across all RoPE types)
            float base_inv_freq = base_inv_frequencies[dim_idx];

            // 2. Polymorphic scaling resolution
            // Passing pre-computed base_inv_freq avoids redundant pow() calculations in subclasses
            float freq_scale = scaling_ ? scaling_->get_freq_scale(pos, dim_idx, base_inv_freq) : 1.0f;
            float mag_scale = scaling_ ? scaling_->get_magnitude_scale(pos, dim_idx, base_inv_freq) : 1.0f;

            // 3. Compute final angle and sin/cos values
            float angle = static_cast<float>(pos) * base_inv_freq * freq_scale;

            sin_data[pos * cache_dim + dim_idx] = std::sin(angle) * mag_scale;
            cos_data[pos * cache_dim + dim_idx] = std::cos(angle) * mag_scale;
        }
    }

    auto upload_typed_cache = [&](auto &sin_values, auto &cos_values) {
        if (cos_sin_cache_next) {
            auto combined = combine_cos_sin_cache(cos_values, sin_values, max_seq_len_, cache_dim);
            auto combined_cpu = Tensor::from_blob(combined.data(), {max_seq_len_, rotary_dim_}, dtype_, cpu_device);
            upload_cache(runtime, cos_sin_cache_next, combined_cpu);
        } else {
            auto sin_cpu = Tensor::from_blob(sin_values.data(), {max_seq_len_, cache_dim}, dtype_, cpu_device);
            auto cos_cpu = Tensor::from_blob(cos_values.data(), {max_seq_len_, cache_dim}, dtype_, cpu_device);
            upload_cache(runtime, sin_cache_next, sin_cpu);
            upload_cache(runtime, cos_cache_next, cos_cpu);
        }
    };

    // Convert to target dtype on CPU (matching Python's numpy astype conversion pattern)
    // Python: np_array.astype(ml_dtypes.bfloat16, copy=True) converts F32 -> BF16
    if (dtype_ == DataType::kFloat32) {
        // Direct use of F32 data
        upload_typed_cache(sin_data, cos_data);
    } else if (dtype_ == DataType::kBFloat16) {
        // Convert F32 to BF16 using the same conversion as Python's ml_dtypes.bfloat16
        // This uses round-to-nearest-even (matching _f32_to_bf16 implementation)
        std::vector<bf16_t> sin_bf16_data(max_seq_len_ * cache_dim);
        std::vector<bf16_t> cos_bf16_data(max_seq_len_ * cache_dim);

        for (size_t i = 0; i < sin_data.size(); i++) {
            sin_bf16_data[i] = ::utils::cast<bf16_t, float>(sin_data[i]);
            cos_bf16_data[i] = ::utils::cast<bf16_t, float>(cos_data[i]);
        }

        upload_typed_cache(sin_bf16_data, cos_bf16_data);
    } else if (dtype_ == DataType::kFloat16) {
        // Convert F32 to F16
        std::vector<fp16_t> sin_f16_data(max_seq_len_ * cache_dim);
        std::vector<fp16_t> cos_f16_data(max_seq_len_ * cache_dim);

        for (size_t i = 0; i < sin_data.size(); i++) {
            sin_f16_data[i] = ::utils::cast<fp16_t, float>(sin_data[i]);
            cos_f16_data[i] = ::utils::cast<fp16_t, float>(cos_data[i]);
        }

        upload_typed_cache(sin_f16_data, cos_f16_data);
    } else {
        throw std::runtime_error(
            "RoPE cache dtype conversion not yet supported for dtype: "
            + std::to_string(static_cast<int>(dtype_)));
    }
    // The source vectors above are stack-owned. Complete their asynchronous
    // device copies before they go out of scope; this runs only at creation.
    runtime.syncStream();
    if (cos_sin_cache_next) {
        cos_cache_next = Tensor(cos_sin_cache_next->narrow({{1, 0, cache_dim}}));
        sin_cache_next = Tensor(cos_sin_cache_next->narrow({{1, cache_dim, cache_dim}}));
    }
    // Publish only complete tables. The factory caches only a completed constructor.
    sin_cache_ = sin_cache_next;
    cos_cache_ = cos_cache_next;
    cos_sin_cache_ = cos_sin_cache_next;



}

Tensor RoPE::forward(Runtime &runtime, const Tensor &x, const Tensor &pos, bool in_place) const {
    runtime.assertOwnerThread();
    if (cache_runtime_owner_.get() != &runtime)
        throw std::invalid_argument("RoPE cache belongs to another Runtime owner");
    if (runtime.device() != device_ || x->device() != device_ || pos->device() != device_) {
        throw std::invalid_argument("RoPE Runtime, input, positions and cache devices must agree");
    }
    if (mrope_section_.has_value()) {
        throw std::runtime_error("MRoPE single-tensor forward is not implemented; use fused forward(q, k, positions) instead");
    }
    Tensor y;
    if (in_place) {
        y = Tensor(x);
    } else {
        y = Tensor::empty(runtime, x->shape(), x->dtype());
        if (cos_sin_cache_ || rotary_dim_ < head_dim_) {
            if (device_.type() == Device::Type::kCpu) {
                y->copy_from(runtime, x);
            } else {
                infinilm::execution::submit<infini::ops::Copy>(
                    runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                    x, false, y);
            }
        }
    }

    if (cos_sin_cache_) {
        execution::submit<infini::ops::RotaryEmbedding>(
            runtime, execution::RecordingPolicy::AllowHostReplay,
            infini::ops::RotaryEmbedding::DefaultConfig(runtime.device().type()),
            pos, y, std::optional<Tensor>{}, cos_sin_cache_,
            static_cast<int64_t>(
                x->ndim() == pos->ndim() + 2
                    ? x->size(x->ndim() - 1)
                    : head_dim_),
            algo_ == Algo::GPT_NEOX, int64_t{0}, false);
        INFINILM_TRACE_OP(RotaryEmbedding, y);
        return y;
    }

    size_t ndim = x->ndim();
    auto rope_output = y->narrow({{ndim - 1, 0, rotary_dim_}});
    auto rope_input = x->narrow({{ndim - 1, 0, rotary_dim_}});
    execution::submit<infini::ops::SeparateTableRope>(
        runtime, execution::RecordingPolicy::HostOnly,
        infini::ops::SeparateTableRope::DefaultConfig(runtime.device().type()),
        rope_input, pos, sin_cache_, cos_cache_, algo_ == Algo::GPT_NEOX, rope_output);
    INFINILM_TRACE_OP(RoPE, rope_output);
    return y;
}

static Tensor mrope_flatten_input(const Tensor &x, size_t head_dim, const char *name) {
    if (x->ndim() == 2) {
        if (x->size(1) % head_dim != 0) {
            throw std::runtime_error(std::string("MRoPE expects ") + name + " hidden size to be a multiple of head_dim");
        }
        return x;
    }
    if (x->ndim() == 3 && x->size(2) == head_dim) {
        return x->view({x->size(0), x->size(1) * head_dim});
    }
    throw std::runtime_error(std::string("MRoPE expects ") + name + " with shape [num_tokens, num_heads * head_dim] or [num_tokens, num_heads, head_dim]");
}

static Tensor mrope_flatten_output(const Tensor &x, size_t head_dim, const char *name) {
    if (x->ndim() == 2) {
        if (x->size(1) % head_dim != 0) {
            throw std::runtime_error(std::string("MRoPE expects ") + name + " hidden size to be a multiple of head_dim");
        }
        return x->view({x->size(0), x->size(1)});
    }
    if (x->ndim() == 3 && x->size(2) == head_dim) {
        return x->view({x->size(0), x->size(1) * head_dim});
    }
    throw std::runtime_error(std::string("MRoPE expects ") + name + " with shape [num_tokens, num_heads * head_dim] or [num_tokens, num_heads, head_dim]");
}

std::pair<Tensor, Tensor> RoPE::forward(Runtime &runtime, const Tensor &q, const Tensor &k, const Tensor &positions) const {
    runtime.assertOwnerThread();
    if (cache_runtime_owner_.get() != &runtime)
        throw std::invalid_argument("RoPE cache belongs to another Runtime owner");
    for (const auto &t : {q, k, positions})
        if (!t || t->device() != runtime.device() || runtime.device() != device_)
            throw std::invalid_argument("RoPE Runtime, input, positions and cache devices must agree");
    if (!mrope_section_.has_value()) {
        auto q_out = Tensor::empty(runtime, q->shape(), q->dtype());
        auto k_out = Tensor::empty(runtime, k->shape(), k->dtype());
        return forward(runtime, q_out, k_out, q, k, positions);
    }
    auto q_flat = mrope_flatten_input(q, head_dim_, "q");
    auto k_flat = mrope_flatten_input(k, head_dim_, "k");
    auto q_out = Tensor::empty(runtime, q_flat->shape(), q_flat->dtype());
    auto k_out = Tensor::empty(runtime, k_flat->shape(), k_flat->dtype());
    const auto &section = mrope_section_.value();
    execution::submit<infini::ops::Mrope>(
        runtime, execution::RecordingPolicy::HostOnly,
        infini::ops::Mrope::DefaultConfig(runtime.device().type()),
        q_flat, k_flat, cos_cache_, sin_cache_, positions,
        static_cast<int>(head_dim_), static_cast<int>(rotary_dim_),
        section[0], section[1], section[2], mrope_interleaved_, q_out, k_out);
    INFINILM_TRACE_OP(MRoPE, q_out);
    return {q_out->view(q->shape()), k_out->view(k->shape())};
}

std::pair<Tensor, Tensor> RoPE::forward(Runtime &runtime, Tensor q_out,
                                        Tensor k_out,
                                        const Tensor &q,
                                        const Tensor &k,
                                        const Tensor &positions) const {
    runtime.assertOwnerThread();
    if (cache_runtime_owner_.get() != &runtime)
        throw std::invalid_argument("RoPE cache belongs to another Runtime owner");
    for (const auto &t : {q_out, k_out, q, k, positions})
        if (!t || t->device() != runtime.device() || runtime.device() != device_)
            throw std::invalid_argument("RoPE Runtime, outputs, inputs, positions and cache devices must agree");
    if (!mrope_section_.has_value()) {
        if (cos_sin_cache_) {
            if (q_out->data() != q->data()) {
                q_out->copy_from(runtime, q);
            }
            if (k_out->data() != k->data()) {
                k_out->copy_from(runtime, k);
            }
            execution::submit<infini::ops::RotaryEmbedding>(
                runtime, execution::RecordingPolicy::AllowHostReplay,
                infini::ops::RotaryEmbedding::DefaultConfig(runtime.device().type()),
                positions, q_out, std::optional<Tensor>{k_out}, cos_sin_cache_,
                static_cast<int64_t>(
                    q_out->ndim() == positions->ndim() + 2
                        ? q_out->size(q_out->ndim() - 1)
                        : head_dim_),
                algo_ == Algo::GPT_NEOX, int64_t{0}, false);
            INFINILM_TRACE_OP(RotaryEmbedding, q_out);
            return {q_out, k_out};
        }
        auto apply_standard = [this, &runtime, &positions](Tensor out, const Tensor &in) {
            if (rotary_dim_ < head_dim_) {
                out->copy_from(runtime, in);
            }
            size_t ndim = in->ndim();
            auto rope_output = out->narrow({{ndim - 1, 0, rotary_dim_}});
            auto rope_input = in->narrow({{ndim - 1, 0, rotary_dim_}});
            execution::submit<infini::ops::SeparateTableRope>(
                runtime, execution::RecordingPolicy::HostOnly,
                infini::ops::SeparateTableRope::DefaultConfig(runtime.device().type()),
                rope_input, positions, sin_cache_, cos_cache_, algo_ == Algo::GPT_NEOX, rope_output);
            INFINILM_TRACE_OP(RoPE, rope_output);
        };
        apply_standard(q_out, q);
        apply_standard(k_out, k);
        return {q_out, k_out};
    }
    auto q_flat = mrope_flatten_input(q, head_dim_, "q");
    auto k_flat = mrope_flatten_input(k, head_dim_, "k");
    auto q_out_flat = mrope_flatten_output(q_out, head_dim_, "q_out");
    auto k_out_flat = mrope_flatten_output(k_out, head_dim_, "k_out");
    const auto &section = mrope_section_.value();
    execution::submit<infini::ops::Mrope>(
        runtime, execution::RecordingPolicy::HostOnly,
        infini::ops::Mrope::DefaultConfig(runtime.device().type()),
        q_flat, k_flat, cos_cache_, sin_cache_, positions,
        static_cast<int>(head_dim_), static_cast<int>(rotary_dim_),
        section[0], section[1], section[2], mrope_interleaved_, q_out_flat, k_out_flat);
    INFINILM_TRACE_OP(MRoPE, q_out_flat);
    return {q_out, k_out};
}

std::string RoPE::extra_repr() const {
    std::string algo_str = (algo_ == Algo::GPT_J) ? "GPT_J" : "GPT_NEOX";
    std::string repr = "RoPE(head_dim=" + std::to_string(head_dim_) + ", rotary_dim=" + std::to_string(rotary_dim_) + ", max_seq_len=" + std::to_string(max_seq_len_) + ", theta=" + std::to_string(theta_) + ", algo=" + algo_str + ", dtype=" + std::to_string(static_cast<int>(dtype_));
    if (mrope_section_.has_value()) {
        const auto &section = mrope_section_.value();
        repr += ", mrope_section=[" + std::to_string(section[0]) + "," + std::to_string(section[1]) + "," + std::to_string(section[2]) + "]";
        repr += ", mrope_interleaved=" + std::string(mrope_interleaved_ ? "true" : "false");
    }
    repr += ")";
    return repr;
}

} // namespace infinilm::layers
