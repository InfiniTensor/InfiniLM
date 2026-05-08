#pragma once

#include "infinicore/device.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <infiniccl.h>
#include <memory>
#include <optional>
#include <vector>

namespace infinilm::backends::ops {

struct GraphTaskUpdate;
using GraphTaskUpdates = std::vector<std::shared_ptr<GraphTaskUpdate>>;

bool should_use(const infinicore::Device &device);

void begin_graph_task_capture();

GraphTaskUpdates end_graph_task_capture();

void update_graph_tasks(const GraphTaskUpdates &updates);

infinicore::Tensor embedding(const infinicore::Tensor &input_ids,
                             const infinicore::Tensor &weight);

infinicore::Tensor linear(const infinicore::Tensor &input,
                          const infinicore::Tensor &weight,
                          std::optional<infinicore::Tensor> bias);

infinicore::Tensor linear_w8a8i8(const infinicore::Tensor &input,
                                 const infinicore::Tensor &weight,
                                 const infinicore::Tensor &weight_scale,
                                 std::optional<infinicore::Tensor> bias);

infinicore::Tensor linear_w4a16_awq(const infinicore::Tensor &input,
                                    const infinicore::Tensor &qweight,
                                    const infinicore::Tensor &scales,
                                    const infinicore::Tensor &qzeros,
                                    std::optional<infinicore::Tensor> bias);

infinicore::Tensor linear_w4a16_gptq_qy(const infinicore::Tensor &input,
                                        const infinicore::Tensor &qweight,
                                        const infinicore::Tensor &qzeros,
                                        const infinicore::Tensor &scales,
                                        int64_t group_size,
                                        int64_t bits);

infinicore::Tensor add(const infinicore::Tensor &a,
                       const infinicore::Tensor &b);

void add_(const infinicore::Tensor &out,
          const infinicore::Tensor &a,
          const infinicore::Tensor &b);

void rms_norm_(const infinicore::Tensor &out,
               const infinicore::Tensor &input,
               const infinicore::Tensor &weight,
               float eps);

infinicore::Tensor rms_norm(const infinicore::Tensor &input,
                            const infinicore::Tensor &weight,
                            float eps);

infinicore::Tensor layer_norm(const infinicore::Tensor &input,
                              const infinicore::Tensor &weight,
                              const infinicore::Tensor &bias,
                              float eps);

void add_rms_norm_(infinicore::Tensor &input,
                   infinicore::Tensor &residual,
                   const infinicore::Tensor &weight,
                   float eps);

infinicore::Tensor swiglu(const infinicore::Tensor &up,
                          const infinicore::Tensor &gate);

void reshape_and_cache(const infinicore::Tensor &key,
                       const infinicore::Tensor &value,
                       const infinicore::Tensor &kv_cache,
                       const infinicore::Tensor &slot_mapping);

void kv_caching_(const infinicore::Tensor &k_cache,
                 const infinicore::Tensor &v_cache,
                 const infinicore::Tensor &k,
                 const infinicore::Tensor &v,
                 const infinicore::Tensor &past_sequence_lengths);

void paged_caching_(const infinicore::Tensor &k_cache,
                    const infinicore::Tensor &v_cache,
                    const infinicore::Tensor &k,
                    const infinicore::Tensor &v,
                    const infinicore::Tensor &slot_mapping);

void rotary_embedding(const infinicore::Tensor &positions,
                      const infinicore::Tensor &query,
                      const infinicore::Tensor &key,
                      const infinicore::Tensor &cos_sin_cache,
                      int64_t head_size,
                      std::optional<infinicore::Tensor> query_out = std::nullopt,
                      std::optional<infinicore::Tensor> key_out = std::nullopt);

void mha_varlen_fwd(const infinicore::Tensor &out,
                    const infinicore::Tensor &q,
                    const infinicore::Tensor &k,
                    const infinicore::Tensor &v,
                    const infinicore::Tensor &cu_seqlens_q,
                    const infinicore::Tensor &cu_seqlens_k,
                    const infinicore::Tensor &block_table,
                    int64_t max_seqlen_q,
                    int64_t max_seqlen_k,
                    float softmax_scale);

void mha_fwd_kvcache(const infinicore::Tensor &out,
                     const infinicore::Tensor &q,
                     const infinicore::Tensor &kcache,
                     const infinicore::Tensor &vcache,
                     const infinicore::Tensor &seqlens_k,
                     const infinicore::Tensor &block_table,
                     float softmax_scale,
                     std::optional<infinicore::Tensor> seqlens_k_host = std::nullopt);

void paged_attention(const infinicore::Tensor &out,
                     const infinicore::Tensor &q,
                     const infinicore::Tensor &kcache,
                     const infinicore::Tensor &vcache,
                     const infinicore::Tensor &seqlens_k,
                     const infinicore::Tensor &block_table,
                     int64_t num_heads,
                     int64_t num_kv_heads,
                     int64_t head_size,
                     float softmax_scale,
                     std::optional<infinicore::Tensor> seqlens_k_host = std::nullopt,
                     std::optional<infinicore::Tensor> block_table_host = std::nullopt);

void paged_attention_prefill_(const infinicore::Tensor &out,
                              const infinicore::Tensor &q,
                              const infinicore::Tensor &kcache,
                              const infinicore::Tensor &vcache,
                              const infinicore::Tensor &block_tables,
                              const infinicore::Tensor &sequence_lengths,
                              const infinicore::Tensor &input_offsets,
                              std::optional<infinicore::Tensor> alibi_slopes,
                              float softmax_scale);

void paged_attention_(const infinicore::Tensor &out,
                      const infinicore::Tensor &q,
                      const infinicore::Tensor &kcache,
                      const infinicore::Tensor &vcache,
                      const infinicore::Tensor &block_tables,
                      const infinicore::Tensor &sequence_lengths,
                      std::optional<infinicore::Tensor> alibi_slopes,
                      float softmax_scale);

infinicore::Tensor flash_attention(const infinicore::Tensor &q,
                                   const infinicore::Tensor &k,
                                   const infinicore::Tensor &v,
                                   const infinicore::Tensor &sequence_lengths,
                                   float softmax_scale,
                                   bool causal);

infinicore::Tensor matmul(const infinicore::Tensor &a,
                          const infinicore::Tensor &b,
                          float scale = 1.0f);

infinicore::Tensor conv2d(const infinicore::Tensor &input,
                          const infinicore::Tensor &weight,
                          const infinicore::Tensor &bias,
                          const std::vector<size_t> &pads,
                          const std::vector<size_t> &strides,
                          const std::vector<size_t> &dilations);

void softmax_(const infinicore::Tensor &out,
              const infinicore::Tensor &input,
              int axis = -1);

infinicore::Tensor gelu(const infinicore::Tensor &input);

infinicore::Tensor gelu_tanh(const infinicore::Tensor &input);

infinicore::Tensor relu(const infinicore::Tensor &input);

void causal_softmax_(const infinicore::Tensor &out,
                     const infinicore::Tensor &input);

void mha_varlen_(const infinicore::Tensor &out,
                 const infinicore::Tensor &q,
                 const infinicore::Tensor &k,
                 const infinicore::Tensor &v,
                 const infinicore::Tensor &cu_seqlens_q,
                 const infinicore::Tensor &cu_seqlens_k,
                 const infinicore::Tensor &block_table,
                 int64_t max_seqlen_q,
                 int64_t max_seqlen_k,
                 std::optional<infinicore::Tensor> alibi_slopes,
                 float softmax_scale);

infinicore::Tensor mha_kvcache(const infinicore::Tensor &q,
                               const infinicore::Tensor &kcache,
                               const infinicore::Tensor &vcache,
                               const infinicore::Tensor &seqlens_k,
                               const infinicore::Tensor &block_table,
                               std::optional<infinicore::Tensor> alibi_slopes,
                               float softmax_scale);

infinicore::Tensor per_tensor_quant_i8(const infinicore::Tensor &input,
                                       const infinicore::Tensor &scale,
                                       const infinicore::Tensor &zero_point,
                                       bool saturate);

void per_tensor_dequant_i8_(const infinicore::Tensor &out,
                            const infinicore::Tensor &input,
                            const infinicore::Tensor &scale,
                            const infinicore::Tensor &zero_point);

infinicore::Tensor sample_from_logits(const infinicore::Tensor &logits,
                                      const infinicore::Tensor &input_offsets,
                                      float temperature,
                                      int top_k,
                                      float top_p);

void random_sample_(const infinicore::Tensor &out,
                    const infinicore::Tensor &score,
                    float random_val,
                    float top_p,
                    int top_k,
                    float temperature);

void allreduce_(const infinicore::Tensor &out,
                const infinicore::Tensor &input,
                infinicclReduceOp_t op,
                infinicclComm_t communicator);

class Embedding : public infinicore::nn::Embedding {
public:
    using infinicore::nn::Embedding::Embedding;

    infinicore::Tensor forward(const infinicore::Tensor &indices) const;
};

class RMSNorm : public infinicore::nn::RMSNorm {
public:
    using infinicore::nn::RMSNorm::RMSNorm;

    infinicore::Tensor forward(const infinicore::Tensor &x) const;
    void forward_inplace(infinicore::Tensor &x, infinicore::Tensor &residual) const;
};

class ReplicatedLinear : public infinicore::nn::Linear {
public:
    using infinicore::nn::Linear::Linear;

    infinicore::Tensor forward(infinicore::Tensor &input) const;
};

class ColumnParallelLinear : public infinicore::nn::ColumnParallelLinear {
public:
    using infinicore::nn::ColumnParallelLinear::ColumnParallelLinear;

    infinicore::Tensor forward(infinicore::Tensor &input) const;
};

class RowParallelLinear : public infinicore::nn::RowParallelLinear {
public:
    using infinicore::nn::RowParallelLinear::RowParallelLinear;

    infinicore::Tensor forward(infinicore::Tensor &input) const;
};

} // namespace infinilm::backends::ops
