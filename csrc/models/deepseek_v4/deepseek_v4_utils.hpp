#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {

std::vector<float> tensor_to_float_vector(const infinicore::Tensor &tensor);
std::vector<int64_t> tensor_to_int64_vector(const infinicore::Tensor &tensor);
bool debug_trace_enabled();
bool debug_trace_layer_enabled(size_t layer_idx);
void debug_trace_tensor(const std::string &name, const infinicore::Tensor &tensor);
void debug_trace_layer_tensor(const std::string &name, size_t layer_idx, const infinicore::Tensor &tensor);
infinicore::Tensor float_vector_to_tensor(const std::vector<float> &values,
                                          const infinicore::Shape &shape,
                                          infinicore::DataType dtype,
                                          const infinicore::Device &device);
infinicore::Tensor int64_vector_to_tensor(const std::vector<int64_t> &values,
                                          const infinicore::Shape &shape,
                                          const infinicore::Device &device);


infinicore::Tensor position_ids_for_rope(const infinicore::Tensor &positions, size_t seq_len);
std::vector<int64_t> position_ids_as_vector(const infinicore::Tensor &pos_ids);

struct DeepseekV4RopeParams {
    size_t head_dim{0};
    size_t rope_dim{0};
    double rope_theta{10000.0};
    bool use_yarn{false};
    double yarn_factor{1.0};
    double yarn_beta_fast{32.0};
    double yarn_beta_slow{1.0};
    int64_t yarn_original_seq_len{0};
    double yarn_extrapolation_factor{1.0};
};

// Partial interleaved RoPE on the trailing rope_dim of each head in x [B, S, H, D].
infinicore::Tensor apply_rotary_pos_emb(const infinicore::Tensor &x,
                                        const std::vector<int64_t> &positions,
                                        const DeepseekV4RopeParams &cfg,
                                        bool inverse = false);

// RoPE on a single head vector already stored in a flat buffer.
void apply_rope_at_offset(std::vector<float> &values,
                          size_t offset,
                          int64_t position,
                          const DeepseekV4RopeParams &cfg,
                          bool inverse = false);

struct DeepseekV4MHCParams {
    std::vector<float> pre;
    std::vector<float> post;
    std::vector<float> comb;
    infinicore::Tensor pre_gpu;
    infinicore::Tensor post_gpu;
    infinicore::Tensor comb_gpu;
    bool gpu_valid{false};
    size_t batch_size{0};
    size_t seq_len{0};
    size_t hc_mult{0};
};

// GPU-resident static mHC tensors reused across forwards (fn^T for matmul).
struct DeepseekV4MHCGpuCache {
    infinicore::Tensor fn_mat_right;
    infinicore::Device device;
    infinicore::DataType matmul_dtype{infinicore::DataType::F32};
    size_t mix_hc{0};
    size_t flat_dim{0};
    bool valid{false};
};

// Cached CPU copy of static mHC weights (base/fn/scale); safe to reuse across forwards.
struct DeepseekV4MHCCoeffs {
    std::vector<float> base;
    std::vector<float> fn;
    float scale[3]{1.0f, 1.0f, 1.0f};
    size_t mix_hc{0};
    size_t flat_dim{0};
    mutable DeepseekV4MHCGpuCache gpu;
};

void ensure_mhc_coeffs_cached(DeepseekV4MHCCoeffs &cache,
                              const infinicore::Tensor &base,
                              const infinicore::Tensor &fn,
                              const infinicore::Tensor &scale,
                              size_t hc_mult,
                              size_t hidden_size);

struct DeepseekV4MHCPrepareResult {
    DeepseekV4MHCParams params;
    infinicore::Tensor collapsed;
};

DeepseekV4MHCParams build_mhc_params(const infinicore::Tensor &x,
                                     const infinicore::Tensor &base,
                                     const infinicore::Tensor &fn,
                                     const infinicore::Tensor &scale,
                                     size_t hc_mult,
                                     size_t hidden_size,
                                     size_t sinkhorn_iters,
                                     double eps);

DeepseekV4MHCPrepareResult mhc_prepare(const infinicore::Tensor &x,
                                       const infinicore::Tensor &base,
                                       const infinicore::Tensor &fn,
                                       const infinicore::Tensor &scale,
                                       DeepseekV4MHCCoeffs &coeffs_cache,
                                       size_t hc_mult,
                                       size_t hidden_size,
                                       size_t sinkhorn_iters,
                                       double eps);

infinicore::Tensor mhc_collapse(const infinicore::Tensor &x,
                                const DeepseekV4MHCParams &params);

infinicore::Tensor mhc_pre(const infinicore::Tensor &x,
                           const DeepseekV4MHCParams &params);

infinicore::Tensor mhc_post(const infinicore::Tensor &new_x,
                            const infinicore::Tensor &residual,
                            const DeepseekV4MHCParams &params);

infinicore::Tensor expand_hc_stream(const infinicore::Tensor &hidden_states,
                                    size_t hc_mult);

infinicore::Tensor mhc_head_pre(const infinicore::Tensor &x,
                                const infinicore::Tensor &base,
                                const infinicore::Tensor &fn,
                                const infinicore::Tensor &scale,
                                DeepseekV4MHCCoeffs &coeffs_cache,
                                size_t hc_mult,
                                size_t hidden_size,
                                double eps);

} // namespace infinilm::models::deepseek_v4
