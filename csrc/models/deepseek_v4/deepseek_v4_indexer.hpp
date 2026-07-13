#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "deepseek_v4_compressor.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Indexer : public infinicore::nn::Module {
public:
    DeepseekV4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t compress_ratio,
                      const infinicore::Device &device);

    infinicore::Tensor forward_tensor(const infinicore::Tensor &hidden_states,
                                      const infinicore::Tensor &q_residual,
                                      const std::vector<int64_t> &positions,
                                      size_t &top_k,
                                      size_t query_start = 0,
                                      size_t query_len = 0) const;

private:
    void reset_runtime_state() const override;

    infinicore::Tensor get_or_update_compressed_cache(
        const infinicore::Tensor &hidden_states,
        size_t batch_size,
        size_t expected_blocks,
        size_t query_len) const;

    void set_compressed_cache(const infinicore::Tensor &compressed,
                              size_t batch_size,
                              size_t num_blocks) const;

    void append_compressed_cache(const infinicore::Tensor &new_blocks) const;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wq_b);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, weights_proj);
    INFINICORE_NN_MODULE(DeepseekV4Compressor, compressor);

    size_t hidden_size_{0};
    size_t compress_ratio_{0};
    size_t index_head_dim_{0};
    size_t index_n_heads_{0};
    size_t index_topk_{0};

    mutable infinicore::Tensor compressed_cache_;
    mutable infinicore::Tensor compressed_cache_storage_;
    mutable size_t compressed_cache_batch_{0};
    mutable size_t compressed_cache_blocks_{0};
    mutable size_t compressed_cache_capacity_{0};
};

} // namespace infinilm::models::deepseek_v4
