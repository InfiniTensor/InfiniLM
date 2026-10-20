#pragma once

#include "base_cache.hpp"

#include "runtime/device.hpp"
#include "runtime/tensor.hpp"
#include <cstddef>
#include <runtime/dtype.hpp>
#include <utility>

namespace infinilm::runtime { class Runtime; }
namespace infinilm::engine::distributed { struct RankInfo; }

namespace infinilm::cache {

class MambaCache {
public:
    static infinilm::runtime::Tensor create_layer_conv_state(
        infinilm::runtime::Runtime &runtime,
        const engine::distributed::RankInfo &rank,
        infinilm::runtime::Size k_dim,
        infinilm::runtime::Size v_dim,
        infinilm::runtime::Size num_k_heads,
        infinilm::runtime::Size num_v_heads,
        infinilm::runtime::Size conv_kernel_dim,
        infinilm::runtime::DataType dtype,
        size_t pool_size);

    static infinilm::runtime::Tensor create_layer_ssm_state(
        infinilm::runtime::Runtime &runtime,
        const engine::distributed::RankInfo &rank,
        infinilm::runtime::Size k_dim,
        infinilm::runtime::Size v_dim,
        infinilm::runtime::Size num_k_heads,
        infinilm::runtime::Size num_v_heads,
        infinilm::runtime::DataType dtype,
        size_t pool_size);

private:
    static std::pair<size_t, size_t> get_rank_head_counts(
        infinilm::runtime::Size num_k_heads,
        infinilm::runtime::Size num_v_heads,
        size_t tp_size);
};

} // namespace infinilm::cache
