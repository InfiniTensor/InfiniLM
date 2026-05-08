#pragma once

#include "../tensor.hpp"

namespace infinicore::nn {
class Parameter : public Tensor {
public:
    Parameter();

    Parameter(const Tensor &tensor,
              Size tp_dim = 0,
              Size tp_rank = 0,
              Size tp_size = 1,
              Size num_shards = 0);

    Parameter(const Shape &shape,
              const DataType &dtype,
              const Device &device,
              Size tp_dim = 0,
              Size tp_rank = 0,
              Size tp_size = 1,
              Size num_shards = 0);

    Parameter(const Parameter &other);

    void load_blob(const void *data);

    void load(const Tensor &tensor);

protected:
    // Tensor parallel configs
    Size tp_dim_;         // dimension partitioned
    Size tp_rank_;        // rank of this partition among tp group
    Size tp_size_;        // total number of partitions
    Size num_shards_ = 0; // number of logical shards, used when tp_size > num_kv_head
};
} // namespace infinicore::nn
