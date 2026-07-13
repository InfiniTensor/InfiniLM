#include "deepseek_v4_attention_state.hpp"

#include <algorithm>

namespace infinilm::models::deepseek_v4 {

void DeepseekV4AttentionState::reset() {
    seq_len = 0;
    positions_contiguous = true;
    positions.clear();
    hidden_states.reset();
    q_residual.reset();
    key_states.reset();
    hidden_states_storage_.reset();
    q_residual_storage_.reset();
    key_states_storage_.reset();
    storage_capacity_ = 0;
    kv_comp.reset();
    kv_comp_blocks = 0;
    kv_comp_batch = 0;
    kv_comp_storage_.reset();
    kv_comp_capacity_ = 0;
    block_positions.reset();
    block_positions_blocks = 0;
}

void DeepseekV4AttentionState::initialize(
    const infinicore::Tensor &new_hidden_states,
    const infinicore::Tensor &new_q_residual,
    const infinicore::Tensor &new_key_states,
    const std::vector<int64_t> &new_positions) {
    hidden_states = new_hidden_states;
    q_residual = new_q_residual;
    key_states = new_key_states;
    positions = new_positions;
    seq_len = new_hidden_states->shape()[1];
    positions_contiguous = true;
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] != positions[i - 1] + 1) {
            positions_contiguous = false;
            break;
        }
    }

    hidden_states_storage_.reset();
    q_residual_storage_.reset();
    key_states_storage_.reset();
    storage_capacity_ = seq_len;
    kv_comp.reset();
    kv_comp_blocks = 0;
    kv_comp_batch = 0;
    kv_comp_storage_.reset();
    kv_comp_capacity_ = 0;
    block_positions.reset();
    block_positions_blocks = 0;
}

void DeepseekV4AttentionState::append(
    const infinicore::Tensor &new_hidden_states,
    const infinicore::Tensor &new_q_residual,
    const infinicore::Tensor &new_key_states,
    const std::vector<int64_t> &new_positions) {
    const size_t old_len = seq_len;
    const size_t append_len = new_hidden_states->shape()[1];
    if (append_len == 0) {
        return;
    }
    const size_t new_len = old_len + append_len;
    if (positions_contiguous && !new_positions.empty()) {
        if ((!positions.empty() && new_positions.front() != positions.back() + 1)) {
            positions_contiguous = false;
        }
        for (size_t i = 1; positions_contiguous && i < new_positions.size(); ++i) {
            if (new_positions[i] != new_positions[i - 1] + 1) {
                positions_contiguous = false;
            }
        }
    }

    if (!hidden_states || !q_residual || !key_states) {
        hidden_states = new_hidden_states;
        q_residual = new_q_residual;
        key_states = new_key_states;
        positions.insert(positions.end(), new_positions.begin(), new_positions.end());
        seq_len = positions.size();
        storage_capacity_ = seq_len;
        return;
    }

    if (storage_capacity_ < new_len || !hidden_states_storage_
        || !q_residual_storage_ || !key_states_storage_) {
        size_t new_capacity = storage_capacity_ == 0 ? new_len : storage_capacity_ * 2;
        new_capacity = std::max(new_capacity, old_len + static_cast<size_t>(16));
        new_capacity = std::max(new_capacity, new_len);

        auto grow_storage = [&](const infinicore::Tensor &current,
                                infinicore::Tensor &storage) {
            auto shape = current->shape();
            shape[1] = new_capacity;
            auto next_storage = infinicore::Tensor::empty(
                shape, current->dtype(), current->device());
            if (old_len > 0) {
                auto dst = next_storage->narrow({{1, 0, old_len}});
                dst->copy_from(current);
            }
            storage = next_storage;
        };

        grow_storage(hidden_states, hidden_states_storage_);
        grow_storage(q_residual, q_residual_storage_);
        grow_storage(key_states, key_states_storage_);
        storage_capacity_ = new_capacity;
    }

    auto append_to_storage = [&](infinicore::Tensor &view,
                                 const infinicore::Tensor &storage,
                                 const infinicore::Tensor &value) {
        auto dst = storage->narrow({{1, old_len, append_len}});
        dst->copy_from(value);
        view = storage->narrow({{1, 0, new_len}});
    };

    append_to_storage(hidden_states, hidden_states_storage_, new_hidden_states);
    append_to_storage(q_residual, q_residual_storage_, new_q_residual);
    append_to_storage(key_states, key_states_storage_, new_key_states);
    positions.insert(positions.end(), new_positions.begin(), new_positions.end());
    seq_len = positions.size();
}

void DeepseekV4AttentionState::set_compressed_kv(
    const infinicore::Tensor &compressed,
    size_t batch_size,
    size_t num_blocks) {
    kv_comp = compressed;
    kv_comp_batch = batch_size;
    kv_comp_blocks = num_blocks;
    kv_comp_storage_.reset();
    kv_comp_capacity_ = num_blocks;
}

void DeepseekV4AttentionState::append_compressed_kv(
    const infinicore::Tensor &new_blocks) {
    const auto shape = new_blocks->shape();
    const size_t append_blocks = shape[1];
    if (append_blocks == 0) {
        return;
    }
    const size_t batch_size = shape[0];
    const size_t old_blocks = kv_comp_blocks;
    const size_t new_block_count = old_blocks + append_blocks;

    if (!kv_comp || kv_comp_capacity_ < new_block_count || !kv_comp_storage_
        || kv_comp_batch != batch_size) {
        size_t new_capacity = std::max(new_block_count, old_blocks + static_cast<size_t>(16));
        new_capacity = std::max(new_capacity, kv_comp_capacity_ * 2);
        const size_t head_dim = shape[2];
        std::vector<size_t> storage_shape{batch_size, new_capacity, head_dim};
        auto next_storage = infinicore::Tensor::empty(
            storage_shape, new_blocks->dtype(), new_blocks->device());
        if (kv_comp && old_blocks > 0 && kv_comp_batch == batch_size) {
            next_storage->narrow({{1, 0, old_blocks}})->copy_from(kv_comp);
        }
        kv_comp_storage_ = next_storage;
        kv_comp_capacity_ = new_capacity;
    }

    kv_comp_storage_->narrow({{1, old_blocks, append_blocks}})->copy_from(
        new_blocks);
    kv_comp = kv_comp_storage_->narrow({{1, 0, new_block_count}});
    kv_comp_blocks = new_block_count;
    kv_comp_batch = batch_size;
}

} // namespace infinilm::models::deepseek_v4
