#pragma once
#include "../../engine/distributed/communication_group.hpp"
#include "../../weights/weights.hpp"
#include "dense.hpp"

namespace infinilm::layers::linear {
// Disjoint vocabulary columns plus zeros restore the public full-logits view
// using the existing graph-capable AllReduce submission contract.
class VocabParallelHead : public infinilm::weights::ModelWeights {
public:
    VocabParallelHead(infinilm::runtime::Runtime &runtime, size_t hidden, size_t vocab,
                      infinilm::runtime::DataType dtype, const engine::distributed::RankInfo &rank)
        : vocab_(vocab), rank_(rank.tp_rank), size_(rank.tp_size), comm_(rank.comm) {
        if (size_ < 2 || rank_ >= size_ || vocab % size_ || !comm_) {
            throw std::invalid_argument("vocabulary head TP geometry is invalid");
        }
        weight_ = weights::Weight(runtime, {vocab, hidden}, dtype, 0, rank_, size_);
        bind_weight("weight", weight_);
    }
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input) const {
        auto shape = input->shape();
        shape.back() = vocab_;
        auto output = infinilm::runtime::Tensor::empty(runtime, shape, input->dtype());
        execution::submit<infini::ops::Fill>(runtime,
                                             infini::ops::Fill::DefaultConfig(runtime.device().type()), output, 0.0, output);
        auto local = output->narrow({{output->ndim() - 1, rank_ * (vocab_ / size_), vocab_ / size_}});
        dense_projection(runtime, input, weight_.tensor(), std::nullopt, 1.0F,
                         std::nullopt, false, local);
        engine::distributed::allreduce_(runtime, output, output, infinicclSum, comm_);
        // Sampling may first-load a CUDA module. Do not leave a collective
        // pending on a peer while entering that runtime initialization path.
        if (!runtime.isGraphRecording()) {
            runtime.syncStream();
        }
        return output;
    }

private:
    weights::Weight weight_;
    size_t vocab_, rank_, size_;
    infinicclComm_t comm_;
};
} // namespace infinilm::layers::linear
