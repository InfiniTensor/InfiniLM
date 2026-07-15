#pragma once

#include "infinicore/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::minicpm5_moe::router_cpu_detail {

inline uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(uint32_t));
    return static_cast<uint16_t>(u >> 16);
}

inline uint16_t f32_to_f16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(uint32_t));
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x007FFFFFu;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    uint16_t out = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    return out;
}

inline float bf16_to_f32(uint16_t x) {
    uint32_t u = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

inline float f16_to_f32(uint16_t x) {
    uint32_t sign = (x & 0x8000u) << 16;
    uint32_t exp = (x & 0x7C00u) >> 10;
    uint32_t mant = (x & 0x03FFu);
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            u = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

inline float scalar_to_f32(const infinicore::Tensor &t, size_t idx) {
    auto dtype = t->dtype();
    const std::byte *p = t->data();
    if (dtype == infinicore::DataType::F32) {
        float v;
        std::memcpy(&v, p + idx * sizeof(float), sizeof(float));
        return v;
    }
    if (dtype == infinicore::DataType::BF16) {
        uint16_t v;
        std::memcpy(&v, p + idx * sizeof(uint16_t), sizeof(uint16_t));
        return bf16_to_f32(v);
    }
    if (dtype == infinicore::DataType::F16) {
        uint16_t v;
        std::memcpy(&v, p + idx * sizeof(uint16_t), sizeof(uint16_t));
        return f16_to_f32(v);
    }
    throw std::runtime_error("minicpm5_moe_router_cpu_detail: unsupported scalar dtype");
}

inline float sigmoid_f32(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

inline void topk_indices_desc(const std::vector<float> &vals, size_t k, std::vector<int32_t> &out_idx) {
    const size_t n = vals.size();
    out_idx.resize(k);
    std::vector<int32_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = static_cast<int32_t>(i);
    if (k >= n) {
        out_idx.assign(idx.begin(), idx.end());
        return;
    }
    auto nth = idx.begin() + static_cast<std::ptrdiff_t>(k);
    std::nth_element(idx.begin(), nth, idx.end(), [&](int32_t a, int32_t b) { return vals[a] > vals[b]; });
    out_idx.assign(idx.begin(), nth);
}

struct RouterTopkCpuResult {
    std::vector<std::vector<int32_t>> topk_indices;
    std::vector<std::vector<float>> topk_weights;
};

inline void run_router_topk_cpu(
    const infinicore::Tensor &logits_cpu,
    const infinicore::Tensor &bias_cpu,
    size_t n_tokens,
    size_t n_routed_experts,
    size_t top_k,
    bool norm_topk_prob,
    float routed_scaling_factor,
    size_t n_group,
    size_t topk_group,
    RouterTopkCpuResult &out) {
    const size_t experts_per_group = n_routed_experts / n_group;
    out.topk_indices.resize(n_tokens);
    out.topk_weights.resize(n_tokens);
    std::vector<float> scores_row(n_routed_experts);
    std::vector<float> scores_for_choice(n_routed_experts);
    std::vector<float> group_scores(n_group);
    std::vector<int32_t> chosen_groups;
    std::vector<int32_t> chosen_experts;

    for (size_t t = 0; t < n_tokens; ++t) {
        for (size_t e = 0; e < n_routed_experts; ++e) {
            float logit = scalar_to_f32(logits_cpu, t * n_routed_experts + e);
            scores_row[e] = sigmoid_f32(logit);
        }

        for (size_t e = 0; e < n_routed_experts; ++e) {
            float b = scalar_to_f32(bias_cpu, e);
            scores_for_choice[e] = scores_row[e] + b;
        }

        for (size_t g = 0; g < n_group; ++g) {
            float m1 = -std::numeric_limits<float>::infinity();
            float m2 = -std::numeric_limits<float>::infinity();
            size_t base = g * experts_per_group;
            for (size_t j = 0; j < experts_per_group; ++j) {
                float v = scores_for_choice[base + j];
                if (v > m1) {
                    m2 = m1;
                    m1 = v;
                } else if (v > m2) {
                    m2 = v;
                }
            }
            group_scores[g] = m1 + m2;
        }

        topk_indices_desc(group_scores, topk_group, chosen_groups);

        std::vector<uint8_t> group_keep(n_group, 0);
        for (auto g : chosen_groups) group_keep[static_cast<size_t>(g)] = 1;
        for (size_t g = 0; g < n_group; ++g) {
            if (group_keep[g]) continue;
            size_t base = g * experts_per_group;
            for (size_t j = 0; j < experts_per_group; ++j) {
                scores_for_choice[base + j] = 0.0f;
            }
        }

        topk_indices_desc(scores_for_choice, top_k, chosen_experts);

        out.topk_indices[t] = chosen_experts;
        out.topk_weights[t].resize(top_k);
        float denom = 0.0f;
        for (size_t j = 0; j < top_k; ++j) {
            float w = scores_row[static_cast<size_t>(chosen_experts[j])];
            out.topk_weights[t][j] = w;
            denom += w;
        }
        if (norm_topk_prob) {
            denom += 1e-20f;
            float inv = 1.0f / denom;
            for (size_t j = 0; j < top_k; ++j) out.topk_weights[t][j] *= inv;
        }
        for (size_t j = 0; j < top_k; ++j) out.topk_weights[t][j] *= routed_scaling_factor;
    }
}

inline void write_f32_as_element(infinicore::Tensor t, size_t idx, float v) {
    auto dtype = t->dtype();
    std::byte *p = t->data();
    if (dtype == infinicore::DataType::F32) {
        std::memcpy(p + idx * sizeof(float), &v, sizeof(float));
    } else if (dtype == infinicore::DataType::BF16) {
        uint16_t x = f32_to_bf16(v);
        std::memcpy(p + idx * sizeof(uint16_t), &x, sizeof(uint16_t));
    } else if (dtype == infinicore::DataType::F16) {
        uint16_t x = f32_to_f16(v);
        std::memcpy(p + idx * sizeof(uint16_t), &x, sizeof(uint16_t));
    } else {
        throw std::runtime_error("minicpm5_moe_router_cpu_detail: write_f32_as_element unsupported dtype");
    }
}

} // namespace infinilm::models::minicpm5_moe::router_cpu_detail
