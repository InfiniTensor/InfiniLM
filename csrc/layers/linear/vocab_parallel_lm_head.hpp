#pragma once

#include "base_linear.hpp"

namespace infinilm::layers::linear {

infinicore::Tensor gather_vocab_logits(const infinicore::Tensor &local,
                                       int rank, int world,
                                       infinicclComm_t communicator);

/// Last-dim width written by VocabParallelLMHead::forward for this rank.
/// Root gathers to full vocab; non-root keeps the local shard.
size_t vocab_parallel_logits_bank_width(size_t vocab_size);

class VocabParallelLMHead : public infinilm::nn::BaseLinear {
public:
    VocabParallelLMHead(size_t in_features, size_t vocab_size, bool bias,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device,
                        int rank, int world, infinicclComm_t communicator,
                        int pp_size);

    infinicore::Tensor forward(infinicore::Tensor &input) const;

private:
    int rank_;
    int world_;
    infinicclComm_t communicator_;
};

} // namespace infinilm::layers::linear
