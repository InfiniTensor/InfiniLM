#include "infinicore/nn/rope.hpp"
#include "../../utils/custom_types.h"
#include "../utils.hpp"
#include "infinicore/ops/mrope.hpp"
#include "infinicore/ops/rope.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinicore::nn {

RoPE::RoPE(size_t head_dim,
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
    if (rotary_dim % 2 != 0) {
        throw std::invalid_argument("rotary_dim must be even for RoPE, got " + std::to_string(rotary_dim));
    }
    assert((rotary_dim > 0) && (rotary_dim <= head_dim_));
    if (mrope_section_.has_value()) {
        const auto &section = mrope_section_.value();
        if (section.size() != 3 || section[0] <= 0 || section[1] <= 0 || section[2] <= 0) {
            throw std::invalid_argument("mrope_section must contain 3 positive values");
        }
        if (2 * static_cast<size_t>(section[0] + section[1] + section[2]) != rotary_dim_) {
            throw std::invalid_argument("MRoPE section sum must equal rotary_dim / 2");
        }
    }
    device_ = device;

    // Initialize cache tables
    initialize_cache();
}

void RoPE::initialize_cache() {
    size_t cache_dim = rotary_dim_ / 2;

    // Create sin and cos cache tables: [max_seq_len, cache_dim]
    INFINICORE_NN_BUFFER_INIT(sin_cache, ({max_seq_len_, cache_dim}, dtype_, device_));
    INFINICORE_NN_BUFFER_INIT(cos_cache, ({max_seq_len_, cache_dim}, dtype_, device_));

    // Pre-compute sin and cos values
    // Frequency generation always uses GPT-J style (theta^(-2j/rotary_dim)).
    // The rotation algorithm (algo_) controls how dimensions are paired in the kernel.

    // Compute on CPU first, then copy to device
    auto cpu_device = Device(Device::Type::kCpu, 0);

    // Allocate CPU buffers
    std::vector<float> sin_data(max_seq_len_ * cache_dim);
    std::vector<float> cos_data(max_seq_len_ * cache_dim);

    for (size_t pos = 0; pos < max_seq_len_; pos++) {
        for (size_t dim_idx = 0; dim_idx < cache_dim; dim_idx++) {
            // 1. Base inverse frequency (shared across all RoPE types)
            float base_inv_freq = 1.0f / std::pow(static_cast<float>(theta_), 2.0f * static_cast<float>(dim_idx) / static_cast<float>(rotary_dim_));

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

    // Convert to target dtype on CPU (matching Python's numpy astype conversion pattern)
    // Python: np_array.astype(ml_dtypes.bfloat16, copy=True) converts F32 -> BF16
    if (dtype_ == DataType::kFloat32) {
        // Direct use of F32 data
        auto sin_f32_cpu = Tensor::from_blob(sin_data.data(), {max_seq_len_, cache_dim}, DataType::kFloat32, cpu_device);
        auto cos_f32_cpu = Tensor::from_blob(cos_data.data(), {max_seq_len_, cache_dim}, DataType::kFloat32, cpu_device);
        sin_cache_->copy_from(sin_f32_cpu);
        cos_cache_->copy_from(cos_f32_cpu);
    } else if (dtype_ == DataType::kBFloat16) {
        // Convert F32 to BF16 using the same conversion as Python's ml_dtypes.bfloat16
        // This uses round-to-nearest-even (matching _f32_to_bf16 implementation)
        std::vector<bf16_t> sin_bf16_data(max_seq_len_ * cache_dim);
        std::vector<bf16_t> cos_bf16_data(max_seq_len_ * cache_dim);

        for (size_t i = 0; i < sin_data.size(); i++) {
            sin_bf16_data[i] = utils::cast<bf16_t, float>(sin_data[i]);
            cos_bf16_data[i] = utils::cast<bf16_t, float>(cos_data[i]);
        }

        auto sin_bf16_cpu = Tensor::from_blob(sin_bf16_data.data(), {max_seq_len_, cache_dim}, DataType::kBFloat16, cpu_device);
        auto cos_bf16_cpu = Tensor::from_blob(cos_bf16_data.data(), {max_seq_len_, cache_dim}, DataType::kBFloat16, cpu_device);

        // copy_from handles cross-device copying to target device
        sin_cache_->copy_from(sin_bf16_cpu);
        cos_cache_->copy_from(cos_bf16_cpu);
    } else if (dtype_ == DataType::kFloat16) {
        // Convert F32 to F16
        std::vector<fp16_t> sin_f16_data(max_seq_len_ * cache_dim);
        std::vector<fp16_t> cos_f16_data(max_seq_len_ * cache_dim);

        for (size_t i = 0; i < sin_data.size(); i++) {
            sin_f16_data[i] = utils::cast<fp16_t, float>(sin_data[i]);
            cos_f16_data[i] = utils::cast<fp16_t, float>(cos_data[i]);
        }

        auto sin_f16_cpu = Tensor::from_blob(sin_f16_data.data(), {max_seq_len_, cache_dim}, DataType::kFloat16, cpu_device);
        auto cos_f16_cpu = Tensor::from_blob(cos_f16_data.data(), {max_seq_len_, cache_dim}, DataType::kFloat16, cpu_device);

        sin_cache_->copy_from(sin_f16_cpu);
        cos_cache_->copy_from(cos_f16_cpu);
    } else {
        throw std::runtime_error(
            "RoPE cache dtype conversion not yet supported for dtype: "
            + std::to_string(static_cast<int>(dtype_)));
    }
}

Tensor RoPE::forward(const Tensor &x, const Tensor &pos, bool in_place) const {
    if (mrope_section_.has_value()) {
        throw std::runtime_error("MRoPE single-tensor forward is not implemented; use fused forward(q, k, positions) instead");
    }
    Tensor y;
    if (in_place) {
        y = Tensor(x);
    } else {
        y = Tensor::empty(x->shape(), x->dtype(), x->device());
        if (rotary_dim_ < head_dim_) {
            y->copy_from(x);
        }
    }

    size_t ndim = x->ndim();
    op::rope_(y->narrow({{ndim - 1, 0, rotary_dim_}}),
              x->narrow({{ndim - 1, 0, rotary_dim_}}),
              pos, sin_cache_, cos_cache_, algo_);
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

std::pair<Tensor, Tensor> RoPE::forward(const Tensor &q, const Tensor &k, const Tensor &positions) const {
    if (!mrope_section_.has_value()) {
        auto q_out = Tensor::empty(q->shape(), q->dtype(), q->device());
        auto k_out = Tensor::empty(k->shape(), k->dtype(), k->device());
        return forward(q_out, k_out, q, k, positions);
    }
    auto q_flat = mrope_flatten_input(q, head_dim_, "q");
    auto k_flat = mrope_flatten_input(k, head_dim_, "k");
    auto q_out = Tensor::empty(q_flat->shape(), q_flat->dtype(), q_flat->device());
    auto k_out = Tensor::empty(k_flat->shape(), k_flat->dtype(), k_flat->device());
    const auto &section = mrope_section_.value();
    op::mrope_(q_out,
               k_out,
               q_flat,
               k_flat,
               cos_cache_,
               sin_cache_,
               positions,
               static_cast<int>(head_dim_),
               static_cast<int>(rotary_dim_),
               section[0],
               section[1],
               section[2],
               mrope_interleaved_);
    return {q_out->view(q->shape()), k_out->view(k->shape())};
}

std::pair<Tensor, Tensor> RoPE::forward(const Tensor &q_out,
                                        const Tensor &k_out,
                                        const Tensor &q,
                                        const Tensor &k,
                                        const Tensor &positions) const {
    if (!mrope_section_.has_value()) {
        auto apply_standard = [this, &positions](Tensor out, const Tensor &in) {
            if (rotary_dim_ < head_dim_) {
                out->copy_from(in);
            }
            size_t ndim = in->ndim();
            op::rope_(out->narrow({{ndim - 1, 0, rotary_dim_}}),
                      in->narrow({{ndim - 1, 0, rotary_dim_}}),
                      positions,
                      sin_cache_,
                      cos_cache_,
                      algo_);
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
    op::mrope_(q_out_flat,
               k_out_flat,
               q_flat,
               k_flat,
               cos_cache_,
               sin_cache_,
               positions,
               static_cast<int>(head_dim_),
               static_cast<int>(rotary_dim_),
               section[0],
               section[1],
               section[2],
               mrope_interleaved_);
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

} // namespace infinicore::nn
