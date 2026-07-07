#include "mamba_cache.hpp"

#include "../global_state/global_state.hpp"
#include "infinicore/context/context.hpp"

namespace infinilm::cache {

infinicore::Tensor MambaCache::create_layer_conv_state(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::Size conv_kernel_dim,
    infinicore::DataType dtype,
    size_t pool_size) {
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const auto [num_rank_k_heads, num_rank_v_heads] = get_rank_head_counts(num_k_heads, num_v_heads, rank_info.tp_size);
    const size_t conv_state_len = conv_kernel_dim > 0 ? conv_kernel_dim - 1 : 0;
    const size_t conv_dim = 2 * num_rank_k_heads * k_dim + num_rank_v_heads * v_dim;

    auto conv_state = infinicore::Tensor::zeros(
        {pool_size, conv_dim, conv_state_len},
        dtype,
        rank_info.device);
    infinicore::context::syncStream();
    return conv_state;
}

infinicore::Tensor MambaCache::create_layer_ssm_state(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::DataType dtype,
    size_t pool_size) {
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const auto rank_head_counts = get_rank_head_counts(num_k_heads, num_v_heads, rank_info.tp_size);
    const size_t num_rank_v_heads = rank_head_counts.second;

    auto ssm_state = infinicore::Tensor::zeros(
        {pool_size, num_rank_v_heads, v_dim, k_dim},
        dtype,
        rank_info.device);

    infinicore::context::syncStream();
    return ssm_state;
}

std::pair<size_t, size_t> MambaCache::get_rank_head_counts(
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    size_t tp_size) {
    bool is_kv_replica = (num_k_heads < tp_size && num_v_heads < tp_size && num_k_heads == num_v_heads && tp_size % num_k_heads == 0);
    size_t num_rank_k_heads = is_kv_replica ? 1 : (num_k_heads / tp_size);
    size_t num_rank_v_heads = is_kv_replica ? 1 : (num_v_heads / tp_size);
    return {num_rank_k_heads, num_rank_v_heads};
}

} // namespace infinilm::cache
