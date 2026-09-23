#pragma once

#include "linear.hpp"

namespace infinilm::nn {

// Shares the usual `lm_head.weight` loader and column partitioning. Uneven
// vocabularies remain replicated until the parameter loader supports padding.
class VocabParallelLinear : public ColumnParallelLinear {
public:
    VocabParallelLinear(size_t hidden_size, size_t vocab_size,
                        const infinicore::DataType &dtype, const infinicore::Device &device,
                        size_t tp_rank, size_t tp_size, infinicclComm_t communicator);

    infinicore::Tensor forward(infinicore::Tensor &input) const;
    infinicore::Tensor top_tokens(infinicore::Tensor &input) const;

private:
    infinicclComm_t communicator_;
    infinicore::Tensor vocab_start_;
};

} // namespace infinilm::nn
