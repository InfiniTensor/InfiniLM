#include "deepseek_v4_utils.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>

namespace infinilm::models::deepseek_v4 {
namespace {

size_t numel_from_shape(const infinicore::Shape &shape) {
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

float read_float_at(const std::byte *ptr, infinicore::DataType dtype, size_t idx) {
    switch (dtype) {
    case infinicore::DataType::F32:
        return reinterpret_cast<const float *>(ptr)[idx];
    case infinicore::DataType::F16:
        return f16_to_f32(reinterpret_cast<const uint16_t *>(ptr)[idx]);
    case infinicore::DataType::BF16:
        return bf16_to_f32(reinterpret_cast<const uint16_t *>(ptr)[idx]);
    case infinicore::DataType::I64:
        return static_cast<float>(reinterpret_cast<const int64_t *>(ptr)[idx]);
    case infinicore::DataType::I32:
        return static_cast<float>(reinterpret_cast<const int32_t *>(ptr)[idx]);
    default:
        throw std::runtime_error("DeepseekV4: unsupported tensor dtype for float conversion");
    }
}

void write_float_at(std::byte *ptr, infinicore::DataType dtype, size_t idx, float value) {
    switch (dtype) {
    case infinicore::DataType::F32:
        reinterpret_cast<float *>(ptr)[idx] = value;
        break;
    case infinicore::DataType::F16:
        reinterpret_cast<uint16_t *>(ptr)[idx] = f32_to_f16(value);
        break;
    case infinicore::DataType::BF16:
        reinterpret_cast<uint16_t *>(ptr)[idx] = f32_to_bf16(value);
        break;
    default:
        throw std::runtime_error("DeepseekV4: unsupported tensor dtype for float write");
    }
}

std::vector<float> softmax_with_eps(const std::vector<float> &row, double eps) {
    const float max_value = *std::max_element(row.begin(), row.end());
    std::vector<float> out(row.size());
    double sum = 0.0;
    for (size_t i = 0; i < row.size(); ++i) {
        out[i] = std::exp(row[i] - max_value);
        sum += out[i];
    }
    for (auto &v : out) {
        v = static_cast<float>(v / sum + eps);
    }
    return out;
}

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

float silu(float value) {
    return value * sigmoid(value);
}

} // namespace

std::vector<float> tensor_to_float_vector(const infinicore::Tensor &tensor) {
    auto cpu = tensor->contiguous()->to(infinicore::Device::cpu());
    const size_t count = cpu->numel();
    const auto dtype = cpu->dtype();
    const auto *ptr = cpu->data();
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = read_float_at(ptr, dtype, i);
    }
    return out;
}

std::vector<int64_t> tensor_to_int64_vector(const infinicore::Tensor &tensor) {
    auto cpu = tensor->contiguous()->to(infinicore::Device::cpu());
    const size_t count = cpu->numel();
    const auto dtype = cpu->dtype();
    const auto *ptr = cpu->data();
    std::vector<int64_t> out(count);
    switch (dtype) {
    case infinicore::DataType::I64:
        std::memcpy(out.data(), ptr, count * sizeof(int64_t));
        break;
    case infinicore::DataType::I32:
        for (size_t i = 0; i < count; ++i) {
            out[i] = reinterpret_cast<const int32_t *>(ptr)[i];
        }
        break;
    default:
        throw std::runtime_error("DeepseekV4: unsupported tensor dtype for int64 conversion");
    }
    return out;
}

bool debug_trace_enabled() {
    const char *value = std::getenv("DEEPSEEK_V4_DEBUG_TRACE");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void debug_trace_tensor(const std::string &name, const infinicore::Tensor &tensor) {
    if (!debug_trace_enabled()) {
        return;
    }
    const auto values = tensor_to_float_vector(tensor);
    if (values.empty()) {
        spdlog::info("DSV4_TRACE {} shape=[] mean=0 rms=0 max_abs=0 first=0 last=0", name);
        return;
    }
    double sum = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (float value : values) {
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        max_abs = std::max(max_abs, std::abs(value));
    }
    std::string shape = "[";
    const auto &dims = tensor->shape();
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            shape += ",";
        }
        shape += std::to_string(dims[i]);
    }
    shape += "]";
    spdlog::info("DSV4_TRACE {} shape={} mean={:.9g} rms={:.9g} max_abs={:.9g} first={:.9g} last={:.9g}",
                 name, shape, sum / static_cast<double>(values.size()),
                 std::sqrt(sum_sq / static_cast<double>(values.size())),
                 max_abs, values.front(), values.back());
}

infinicore::Tensor float_vector_to_tensor(const std::vector<float> &values,
                                          const infinicore::Shape &shape,
                                          infinicore::DataType dtype,
                                          const infinicore::Device &device) {
    if (values.size() != numel_from_shape(shape)) {
        throw std::runtime_error("DeepseekV4: float_vector_to_tensor shape mismatch");
    }
    auto cpu = infinicore::Tensor::empty(shape, dtype, infinicore::Device::cpu());
    auto *ptr = cpu->data();
    for (size_t i = 0; i < values.size(); ++i) {
        write_float_at(ptr, dtype, i, values[i]);
    }
    auto out = cpu->to(device);
    if (device.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::syncStream();
    }
    return out;
}

infinicore::Tensor int64_vector_to_tensor(const std::vector<int64_t> &values,
                                          const infinicore::Shape &shape,
                                          const infinicore::Device &device) {
    if (values.size() != numel_from_shape(shape)) {
        throw std::runtime_error("DeepseekV4: int64_vector_to_tensor shape mismatch");
    }
    auto cpu = infinicore::Tensor::empty(shape, infinicore::DataType::I64, infinicore::Device::cpu());
    std::memcpy(cpu->data(), values.data(), values.size() * sizeof(int64_t));
    auto out = cpu->to(device);
    if (device.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::syncStream();
    }
    return out;
}

infinicore::Tensor clamped_swiglu(const infinicore::Tensor &up,
                                  const infinicore::Tensor &gate,
                                  double limit) {
    auto up_values = tensor_to_float_vector(up);
    auto gate_values = tensor_to_float_vector(gate);
    if (up_values.size() != gate_values.size()) {
        throw std::runtime_error("DeepseekV4MLP: up/gate shape mismatch");
    }
    const float lim = static_cast<float>(limit);
    std::vector<float> out(up_values.size());
    for (size_t i = 0; i < out.size(); ++i) {
        const float u = std::max(-lim, std::min(lim, up_values[i]));
        const float g = std::min(gate_values[i], lim);
        out[i] = silu(g) * u;
    }
    return float_vector_to_tensor(out, up->shape(), up->dtype(), up->device());
}

DeepseekV4MHCParams build_mhc_params(const infinicore::Tensor &x,
                                     const infinicore::Tensor &base,
                                     const infinicore::Tensor &fn,
                                     const infinicore::Tensor &scale,
                                     size_t hc_mult,
                                     size_t hidden_size,
                                     size_t sinkhorn_iters,
                                     double eps) {
    const auto shape = x->shape();
    if (shape.size() != 4 || shape[2] != hc_mult || shape[3] != hidden_size) {
        throw std::runtime_error("DeepseekV4MHC: expected x shape [B,S,hc_mult,hidden_size]");
    }
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t flat_dim = hc_mult * hidden_size;
    const size_t mix_hc = (2 + hc_mult) * hc_mult;

    auto x_values = tensor_to_float_vector(x);
    auto base_values = tensor_to_float_vector(base);
    auto fn_values = tensor_to_float_vector(fn);
    auto scale_values = tensor_to_float_vector(scale);
    if (base_values.size() != mix_hc || fn_values.size() != mix_hc * flat_dim || scale_values.size() < 3) {
        throw std::runtime_error("DeepseekV4MHC: parameter shape mismatch");
    }

    DeepseekV4MHCParams params;
    params.batch_size = batch_size;
    params.seq_len = seq_len;
    params.hc_mult = hc_mult;
    params.pre.resize(batch_size * seq_len * hc_mult);
    params.post.resize(batch_size * seq_len * hc_mult);
    params.comb.resize(batch_size * seq_len * hc_mult * hc_mult);

    std::vector<float> flat(flat_dim);
    std::vector<float> mixes(mix_hc);
    std::vector<float> comb_raw(hc_mult * hc_mult);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t token = b * seq_len + s;
            double mean_square = 0.0;
            for (size_t i = 0; i < flat_dim; ++i) {
                flat[i] = x_values[token * flat_dim + i];
                mean_square += static_cast<double>(flat[i]) * flat[i];
            }
            const float rsqrt = static_cast<float>(1.0 / std::sqrt(mean_square / static_cast<double>(flat_dim) + eps));
            for (size_t m = 0; m < mix_hc; ++m) {
                double dot = 0.0;
                const size_t fn_offset = m * flat_dim;
                for (size_t i = 0; i < flat_dim; ++i) {
                    dot += static_cast<double>(fn_values[fn_offset + i]) * flat[i];
                }
                mixes[m] = static_cast<float>(dot * rsqrt);
            }

            const size_t pre_offset = token * hc_mult;
            for (size_t i = 0; i < hc_mult; ++i) {
                params.pre[pre_offset + i] = sigmoid(scale_values[0] * mixes[i] + base_values[i]) + static_cast<float>(eps);
                params.post[pre_offset + i] = 2.0f * sigmoid(scale_values[1] * mixes[hc_mult + i] + base_values[hc_mult + i]);
            }

            for (size_t i = 0; i < hc_mult; ++i) {
                std::vector<float> row(hc_mult);
                for (size_t j = 0; j < hc_mult; ++j) {
                    const size_t idx = 2 * hc_mult + i * hc_mult + j;
                    row[j] = scale_values[2] * mixes[idx] + base_values[idx];
                }
                auto row_sm = softmax_with_eps(row, eps);
                for (size_t j = 0; j < hc_mult; ++j) {
                    comb_raw[i * hc_mult + j] = row_sm[j];
                }
            }

            for (size_t j = 0; j < hc_mult; ++j) {
                double col_sum = eps;
                for (size_t i = 0; i < hc_mult; ++i) {
                    col_sum += comb_raw[i * hc_mult + j];
                }
                for (size_t i = 0; i < hc_mult; ++i) {
                    comb_raw[i * hc_mult + j] = static_cast<float>(comb_raw[i * hc_mult + j] / col_sum);
                }
            }
            for (size_t iter = 1; iter < sinkhorn_iters; ++iter) {
                for (size_t i = 0; i < hc_mult; ++i) {
                    double row_sum = eps;
                    for (size_t j = 0; j < hc_mult; ++j) {
                        row_sum += comb_raw[i * hc_mult + j];
                    }
                    for (size_t j = 0; j < hc_mult; ++j) {
                        comb_raw[i * hc_mult + j] = static_cast<float>(comb_raw[i * hc_mult + j] / row_sum);
                    }
                }
                for (size_t j = 0; j < hc_mult; ++j) {
                    double col_sum = eps;
                    for (size_t i = 0; i < hc_mult; ++i) {
                        col_sum += comb_raw[i * hc_mult + j];
                    }
                    for (size_t i = 0; i < hc_mult; ++i) {
                        comb_raw[i * hc_mult + j] = static_cast<float>(comb_raw[i * hc_mult + j] / col_sum);
                    }
                }
            }

            std::copy(comb_raw.begin(), comb_raw.end(), params.comb.begin() + token * hc_mult * hc_mult);
        }
    }
    return params;
}

infinicore::Tensor mhc_pre(const infinicore::Tensor &x,
                           const DeepseekV4MHCParams &params) {
    const auto shape = x->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hc_mult = shape[2];
    const size_t hidden_size = shape[3];
    auto x_values = tensor_to_float_vector(x);
    std::vector<float> out(batch_size * seq_len * hidden_size, 0.0f);
    for (size_t token = 0; token < batch_size * seq_len; ++token) {
        for (size_t h = 0; h < hc_mult; ++h) {
            const float coeff = params.pre[token * hc_mult + h];
            for (size_t d = 0; d < hidden_size; ++d) {
                out[token * hidden_size + d] += coeff * x_values[(token * hc_mult + h) * hidden_size + d];
            }
        }
    }
    return float_vector_to_tensor(out, {batch_size, seq_len, hidden_size}, x->dtype(), x->device());
}

infinicore::Tensor mhc_post(const infinicore::Tensor &new_x,
                            const infinicore::Tensor &residual,
                            const DeepseekV4MHCParams &params) {
    const auto shape = residual->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hc_mult = shape[2];
    const size_t hidden_size = shape[3];
    auto residual_values = tensor_to_float_vector(residual);
    auto new_values = tensor_to_float_vector(new_x);
    std::vector<float> out(batch_size * seq_len * hc_mult * hidden_size, 0.0f);
    for (size_t token = 0; token < batch_size * seq_len; ++token) {
        for (size_t i = 0; i < hc_mult; ++i) {
            const float post = params.post[token * hc_mult + i];
            for (size_t d = 0; d < hidden_size; ++d) {
                float value = post * new_values[token * hidden_size + d];
                for (size_t j = 0; j < hc_mult; ++j) {
                    value += params.comb[(token * hc_mult + i) * hc_mult + j]
                           * residual_values[(token * hc_mult + j) * hidden_size + d];
                }
                out[(token * hc_mult + i) * hidden_size + d] = value;
            }
        }
    }
    return float_vector_to_tensor(out, residual->shape(), residual->dtype(), residual->device());
}

infinicore::Tensor expand_hc_stream(const infinicore::Tensor &hidden_states,
                                    size_t hc_mult) {
    const auto shape = hidden_states->shape();
    if (shape.size() != 3) {
        throw std::runtime_error("DeepseekV4MHC: expected hidden_states shape [B,S,D]");
    }
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hidden_size = shape[2];

    const auto strides = hidden_states->strides();
    return hidden_states->as_strided(
                            {batch_size, seq_len, hc_mult, hidden_size},
                            {strides[0], strides[1], 0, strides[2]})
        ->contiguous();
}

infinicore::Tensor mhc_head_pre(const infinicore::Tensor &x,
                                const infinicore::Tensor &base,
                                const infinicore::Tensor &fn,
                                const infinicore::Tensor &scale,
                                size_t hc_mult,
                                size_t hidden_size,
                                double eps) {
    const auto shape = x->shape();
    if (shape.size() != 4 || shape[2] != hc_mult || shape[3] != hidden_size) {
        throw std::runtime_error("DeepseekV4MHC: expected x shape [B,S,hc_mult,hidden_size]");
    }
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t flat_dim = hc_mult * hidden_size;
    auto x_values = tensor_to_float_vector(x);
    auto base_values = tensor_to_float_vector(base);
    auto fn_values = tensor_to_float_vector(fn);
    auto scale_values = tensor_to_float_vector(scale);
    const float s = scale_values.empty() ? 1.0f : scale_values[0];
    std::vector<float> out(batch_size * seq_len * hidden_size, 0.0f);
    for (size_t token = 0; token < batch_size * seq_len; ++token) {
        double mean_square = 0.0;
        for (size_t i = 0; i < flat_dim; ++i) {
            const float value = x_values[token * flat_dim + i];
            mean_square += static_cast<double>(value) * value;
        }
        const float rsqrt = static_cast<float>(1.0 / std::sqrt(mean_square / static_cast<double>(flat_dim) + eps));
        std::vector<float> pre(hc_mult);
        for (size_t h = 0; h < hc_mult; ++h) {
            double dot = 0.0;
            for (size_t i = 0; i < flat_dim; ++i) {
                dot += static_cast<double>(fn_values[h * flat_dim + i]) * x_values[token * flat_dim + i];
            }
            pre[h] = sigmoid(s * static_cast<float>(dot * rsqrt) + base_values[h]) + static_cast<float>(eps);
        }
        for (size_t h = 0; h < hc_mult; ++h) {
            for (size_t d = 0; d < hidden_size; ++d) {
                out[token * hidden_size + d] += pre[h] * x_values[(token * hc_mult + h) * hidden_size + d];
            }
        }
    }
    return float_vector_to_tensor(out, {batch_size, seq_len, hidden_size}, x->dtype(), x->device());
}

} // namespace infinilm::models::deepseek_v4
