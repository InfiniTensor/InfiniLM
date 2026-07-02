#pragma once

#include "base_cache.hpp"

#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"
#include <cstddef>
#include <infinicore/dtype.hpp>
#include <utility>

namespace infinilm::cache {

class MambaCache {
public:
    static infinicore::Tensor create_layer_conv_state(
        infinicore::Size k_dim,
        infinicore::Size v_dim,
        infinicore::Size num_k_heads,
        infinicore::Size num_v_heads,
        infinicore::Size conv_kernel_dim,
        infinicore::DataType dtype,
        size_t pool_size);

    static infinicore::Tensor create_layer_ssm_state(
        infinicore::Size k_dim,
        infinicore::Size v_dim,
        infinicore::Size num_k_heads,
        infinicore::Size num_v_heads,
        infinicore::DataType dtype,
        size_t pool_size);

private:
    static std::pair<size_t, size_t> get_rank_head_counts(
        infinicore::Size num_k_heads,
        infinicore::Size num_v_heads,
        size_t tp_size);
};

} // namespace infinilm::cache
