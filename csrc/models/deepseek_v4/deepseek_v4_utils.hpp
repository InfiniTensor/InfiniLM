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
void debug_trace_tensor(const std::string &name, const infinicore::Tensor &tensor);
infinicore::Tensor float_vector_to_tensor(const std::vector<float> &values,
                                          const infinicore::Shape &shape,
                                          infinicore::DataType dtype,
                                          const infinicore::Device &device);
infinicore::Tensor int64_vector_to_tensor(const std::vector<int64_t> &values,
                                          const infinicore::Shape &shape,
                                          const infinicore::Device &device);

infinicore::Tensor clamped_swiglu(const infinicore::Tensor &up,
                                  const infinicore::Tensor &gate,
                                  double limit);

struct DeepseekV4MHCParams {
    std::vector<float> pre;
    std::vector<float> post;
    std::vector<float> comb;
    size_t batch_size{0};
    size_t seq_len{0};
    size_t hc_mult{0};
};

DeepseekV4MHCParams build_mhc_params(const infinicore::Tensor &x,
                                     const infinicore::Tensor &base,
                                     const infinicore::Tensor &fn,
                                     const infinicore::Tensor &scale,
                                     size_t hc_mult,
                                     size_t hidden_size,
                                     size_t sinkhorn_iters,
                                     double eps);

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
                                size_t hc_mult,
                                size_t hidden_size,
                                double eps);

} // namespace infinilm::models::deepseek_v4
