#include "gemma4_rope.hpp"

#include "../../execution/submit.hpp"
#include "runtime/custom_types.h"
#include "runtime/utils.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::gemma4 {
namespace {

using infinilm::runtime::DataType;
using infinilm::runtime::Device;
using infinilm::runtime::Runtime;
using infinilm::runtime::Tensor;

template <typename T>
std::vector<T> pack_cos_sin(const std::vector<T> &cos_vals,
                            const std::vector<T> &sin_vals,
                            size_t max_seq_len,
                            size_t half_dim) {
    std::vector<T> combined(max_seq_len * half_dim * 2);
    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        std::copy_n(cos_vals.begin() + pos * half_dim, half_dim,
                    combined.begin() + pos * half_dim * 2);
        std::copy_n(sin_vals.begin() + pos * half_dim, half_dim,
                    combined.begin() + (pos * 2 + 1) * half_dim);
    }
    return combined;
}

} // namespace

Gemma4RoPE::Gemma4RoPE(Runtime &runtime,
                       size_t head_dim,
                       size_t max_seq_len,
                       double theta,
                       double partial_rotary_factor,
                       bool is_proportional,
                       DataType dtype)
    : device_(runtime.device()),
      runtime_owner_(runtime.shared_from_this()),
      head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      dtype_(dtype) {
    runtime.assertOwnerThread();
    if (head_dim_ == 0 || (head_dim_ % 2) != 0) {
        throw std::invalid_argument("Gemma4RoPE head_dim must be positive and even");
    }
    if (max_seq_len_ == 0 || !std::isfinite(theta) || theta <= 0.0) {
        throw std::invalid_argument("Gemma4RoPE max_seq_len and theta must be positive");
    }

    const size_t half_dim = head_dim_ / 2;
    size_t rope_angles = half_dim;
    if (is_proportional) {
        rope_angles = static_cast<size_t>(std::floor(partial_rotary_factor * static_cast<double>(head_dim_) / 2.0));
        rope_angles = std::min(rope_angles, half_dim);
    }

    // Exact HF formula:
    //   default:      inv_freq[j] = 1 / theta^(2*j / head_dim), j in [0, half_dim)
    //   proportional: inv_freq[j] = 1 / theta^(2*j / head_dim) for j < rope_angles, else 0.0
    std::vector<float> inv_freq(half_dim, 0.0f);
    for (size_t j = 0; j < rope_angles; ++j) {
        const double exponent = (2.0 * static_cast<double>(j)) / static_cast<double>(head_dim_);
        inv_freq[j] = static_cast<float>(1.0 / std::pow(theta, exponent));
    }

    std::vector<float> cos_f32(max_seq_len_ * half_dim);
    std::vector<float> sin_f32(max_seq_len_ * half_dim);
    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        const float p = static_cast<float>(pos);
        for (size_t j = 0; j < half_dim; ++j) {
            const float angle = p * inv_freq[j];
            cos_f32[pos * half_dim + j] = std::cos(angle);
            sin_f32[pos * half_dim + j] = std::sin(angle);
        }
    }

    const bool use_packed_gpu = (device_.type() != Device::Type::kCpu);
    Tensor cos_sin_next;
    Tensor cos_next;
    Tensor sin_next;
    if (use_packed_gpu) {
        cos_sin_next = Tensor::empty_parameter(runtime, {max_seq_len_, head_dim_}, dtype_);
    } else {
        cos_next = Tensor::empty_parameter(runtime, {max_seq_len_, half_dim}, dtype_);
        sin_next = Tensor::empty_parameter(runtime, {max_seq_len_, half_dim}, dtype_);
    }

    const auto cpu_device = Device(Device::Type::kCpu, 0);
    auto upload = [&](const auto &cos_vec, const auto &sin_vec) {
        if (cos_sin_next) {
            auto combined = pack_cos_sin(cos_vec, sin_vec, max_seq_len_, half_dim);
            auto host = Tensor::from_blob(combined.data(), {max_seq_len_, head_dim_}, dtype_, cpu_device);
            cos_sin_next->copy_from(runtime, host);
        } else {
            auto host_cos = Tensor::from_blob(const_cast<typename std::decay_t<decltype(cos_vec)>::value_type *>(cos_vec.data()),
                                              {max_seq_len_, half_dim}, dtype_, cpu_device);
            auto host_sin = Tensor::from_blob(const_cast<typename std::decay_t<decltype(sin_vec)>::value_type *>(sin_vec.data()),
                                              {max_seq_len_, half_dim}, dtype_, cpu_device);
            cos_next->copy_from(runtime, host_cos);
            sin_next->copy_from(runtime, host_sin);
        }
        runtime.syncStream();
    };

    try {
        if (dtype_ == DataType::kFloat32) {
            upload(cos_f32, sin_f32);
        } else if (dtype_ == DataType::kBFloat16) {
            std::vector<bf16_t> cos_bf16(cos_f32.size());
            std::vector<bf16_t> sin_bf16(sin_f32.size());
            for (size_t i = 0; i < cos_f32.size(); ++i) {
                cos_bf16[i] = ::utils::cast<bf16_t, float>(cos_f32[i]);
                sin_bf16[i] = ::utils::cast<bf16_t, float>(sin_f32[i]);
            }
            upload(cos_bf16, sin_bf16);
        } else if (dtype_ == DataType::kFloat16) {
            std::vector<fp16_t> cos_f16(cos_f32.size());
            std::vector<fp16_t> sin_f16(sin_f32.size());
            for (size_t i = 0; i < cos_f32.size(); ++i) {
                cos_f16[i] = ::utils::cast<fp16_t, float>(cos_f32[i]);
                sin_f16[i] = ::utils::cast<fp16_t, float>(sin_f32[i]);
            }
            upload(cos_f16, sin_f16);
        } else {
            throw std::invalid_argument("Gemma4RoPE requires float32, float16, or bfloat16");
        }
    } catch (...) {
        if (device_.type() != Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
        }
        throw;
    }

    if (cos_sin_next) {
        cos_next = Tensor(cos_sin_next->narrow({{1, 0, half_dim}}));
        sin_next = Tensor(cos_sin_next->narrow({{1, half_dim, half_dim}}));
    }
    cos_sin_cache_ = std::move(cos_sin_next);
    cos_cache_ = std::move(cos_next);
    sin_cache_ = std::move(sin_next);
}

Tensor Gemma4RoPE::forward(Runtime &runtime,
                           const Tensor &x,
                           const Tensor &positions,
                           bool in_place) const {
    runtime.assertOwnerThread();
    if (runtime_owner_.get() != &runtime) {
        throw std::invalid_argument("Gemma4RoPE cache belongs to another Runtime owner");
    }
    Tensor y;
    if (in_place) {
        y = Tensor(x);
    } else {
        y = Tensor::empty(runtime, x->shape(), x->dtype());
        if (device_.type() == Device::Type::kCpu) {
            y->copy_from(runtime, x);
        } else {
            infinilm::execution::submit<infini::ops::Copy>(
                runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()),
                x, false, y);
        }
    }

    if (cos_sin_cache_) {
        infinilm::execution::submit<infini::ops::RotaryEmbedding>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::RotaryEmbedding::DefaultConfig(runtime.device().type()),
            positions, y, std::optional<Tensor>{}, cos_sin_cache_,
            static_cast<int64_t>(head_dim_), true /* is_neox */, int64_t{0}, false);
        INFINILM_TRACE_OP(RotaryEmbedding, y);
        return y;
    }

    infinilm::execution::submit<infini::ops::SeparateTableRope>(
        runtime, infinilm::execution::RecordingPolicy::HostOnly,
        infini::ops::SeparateTableRope::DefaultConfig(runtime.device().type()),
        x, positions, sin_cache_, cos_cache_, true /* is_neox */, y);
    INFINILM_TRACE_OP(RoPE, y);
    return y;
}

} // namespace infinilm::models::gemma4
