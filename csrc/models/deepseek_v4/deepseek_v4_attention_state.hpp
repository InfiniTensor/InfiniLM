#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infinilm::models::deepseek_v4 {

enum class DeepseekV4AttentionMode {
    Prefill,
    Decode,
};

struct DeepseekV4AttentionStep {
    DeepseekV4AttentionMode mode{DeepseekV4AttentionMode::Prefill};
    infinicore::Tensor raw_positions;
    std::vector<int64_t> positions;
    size_t query_start{0};
    size_t query_len{0};

    bool is_decode() const { return mode == DeepseekV4AttentionMode::Decode; }
};

class DeepseekV4AttentionState {
public:
    void reset();

    void initialize(const infinicore::Tensor &hidden_states,
                    const infinicore::Tensor &q_residual,
                    const infinicore::Tensor &key_states,
                    const std::vector<int64_t> &positions);

    void append(const infinicore::Tensor &hidden_states,
                const infinicore::Tensor &q_residual,
                const infinicore::Tensor &key_states,
                const std::vector<int64_t> &positions);

    void set_compressed_kv(const infinicore::Tensor &compressed,
                           size_t batch_size,
                           size_t num_blocks);

    void append_compressed_kv(const infinicore::Tensor &new_blocks);

    size_t seq_len{0};
    bool positions_contiguous{true};
    std::vector<int64_t> positions;
    infinicore::Tensor hidden_states;
    infinicore::Tensor q_residual;
    infinicore::Tensor key_states;

    infinicore::Tensor kv_comp;
    size_t kv_comp_blocks{0};
    size_t kv_comp_batch{0};
    infinicore::Tensor block_positions;
    size_t block_positions_blocks{0};

private:
    infinicore::Tensor hidden_states_storage_;
    infinicore::Tensor q_residual_storage_;
    infinicore::Tensor key_states_storage_;
    size_t storage_capacity_{0};
    infinicore::Tensor kv_comp_storage_;
    size_t kv_comp_capacity_{0};
};

} // namespace infinilm::models::deepseek_v4
