#include "mamba_cache.hpp"
#include "../execution/submit.hpp"

#include "../engine/distributed/communication_group.hpp"
#include "kv_cache.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::cache {

infinilm::runtime::Tensor MambaCache::create_layer_conv_state(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    infinilm::runtime::Size k_dim,
    infinilm::runtime::Size v_dim,
    infinilm::runtime::Size num_k_heads,
    infinilm::runtime::Size num_v_heads,
    infinilm::runtime::Size conv_kernel_dim,
    infinilm::runtime::DataType dtype,
    size_t pool_size) {
    validate_cache_runtime(runtime, rank);
    const auto [num_rank_k_heads, num_rank_v_heads] = get_rank_head_counts(num_k_heads, num_v_heads, static_cast<size_t>(rank.tp_size));
    const size_t conv_state_len = conv_kernel_dim > 0 ? conv_kernel_dim - 1 : 0;
    const size_t conv_dim = 2 * num_rank_k_heads * k_dim + num_rank_v_heads * v_dim;

    auto conv_state = infinilm::runtime::Tensor::empty(
        runtime,
        {pool_size, conv_dim, conv_state_len},
        dtype);
    try {
        if (conv_state->nbytes() != 0) {
            if (conv_state->is_contiguous() && !runtime.isGraphRecording()) {
                if (!infini::ops::Fill::valid_output(infinilm::execution::detail::view(conv_state))) {
                    throw std::invalid_argument("zero-fill output layout or storage is invalid");
                }
                runtime.setDeviceMemoryAsync(conv_state->data(), 0, conv_state->nbytes(), runtime.stream());
            } else {
                infinilm::execution::submit<infini::ops::Fill>(
                    runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
                    conv_state, 0.0, conv_state);
            }
            INFINILM_TRACE_OP(Zeros, conv_state);
        }
        runtime.syncStream();
    } catch (...) {
        const auto primary = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        std::rethrow_exception(primary);
    }
    return conv_state;
}

infinilm::runtime::Tensor MambaCache::create_layer_ssm_state(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    infinilm::runtime::Size k_dim,
    infinilm::runtime::Size v_dim,
    infinilm::runtime::Size num_k_heads,
    infinilm::runtime::Size num_v_heads,
    infinilm::runtime::DataType dtype,
    size_t pool_size) {
    validate_cache_runtime(runtime, rank);
    const auto rank_head_counts = get_rank_head_counts(num_k_heads, num_v_heads, static_cast<size_t>(rank.tp_size));
    const size_t num_rank_v_heads = rank_head_counts.second;

    auto ssm_state = infinilm::runtime::Tensor::empty(
        runtime,
        {pool_size, num_rank_v_heads, v_dim, k_dim},
        dtype);
    try {
        if (ssm_state->nbytes() != 0) {
            if (ssm_state->is_contiguous() && !runtime.isGraphRecording()) {
                if (!infini::ops::Fill::valid_output(infinilm::execution::detail::view(ssm_state))) {
                    throw std::invalid_argument("zero-fill output layout or storage is invalid");
                }
                runtime.setDeviceMemoryAsync(ssm_state->data(), 0, ssm_state->nbytes(), runtime.stream());
            } else {
                infinilm::execution::submit<infini::ops::Fill>(
                    runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
                    ssm_state, 0.0, ssm_state);
            }
            INFINILM_TRACE_OP(Zeros, ssm_state);
        }
        runtime.syncStream();
    } catch (...) {
        const auto primary = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        std::rethrow_exception(primary);
    }
    return ssm_state;
}

std::pair<size_t, size_t> MambaCache::get_rank_head_counts(
    infinilm::runtime::Size num_k_heads,
    infinilm::runtime::Size num_v_heads,
    size_t tp_size) {
    if (tp_size == 0 || num_k_heads == 0 || num_v_heads == 0) {
        throw std::invalid_argument("Mamba cache requires positive TP and head counts");
    }
    bool is_kv_replica = (num_k_heads < tp_size && num_v_heads < tp_size && num_k_heads == num_v_heads && tp_size % num_k_heads == 0);
    if (!is_kv_replica && (num_k_heads % tp_size != 0 || num_v_heads % tp_size != 0)) {
        throw std::invalid_argument("Mamba cache head counts must be divisible by TP");
    }
    size_t num_rank_k_heads = is_kv_replica ? 1 : (num_k_heads / tp_size);
    size_t num_rank_v_heads = is_kv_replica ? 1 : (num_v_heads / tp_size);
    return {num_rank_k_heads, num_rank_v_heads};
}

} // namespace infinilm::cache
