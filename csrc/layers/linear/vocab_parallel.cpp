#include "vocab_parallel.hpp"

#include <cstdint>
#include <infinicore/ops/distributed/allgather.hpp>
#include <infinicore/ops/take.hpp>

namespace infinilm::nn {

VocabParallelLinear::VocabParallelLinear(size_t hidden_size, size_t vocab_size,
                                         const infinicore::DataType &dtype, const infinicore::Device &device,
                                         size_t tp_rank, size_t tp_size, infinicclComm_t communicator)
    : ColumnParallelLinear(hidden_size, vocab_size, false, dtype, device,
                           vocab_size % tp_size == 0 ? tp_rank : 0,
                           vocab_size % tp_size == 0 ? tp_size : 1),
      communicator_(communicator) {
    auto start = infinicore::Tensor::empty({1}, infinicore::DataType::I64, infinicore::Device::cpu());
    *reinterpret_cast<int64_t *>(start->data()) = static_cast<int64_t>(tp_rank_ * (vocab_size / tp_size_));
    vocab_start_ = start->to(device);
}

infinicore::Tensor VocabParallelLinear::forward(infinicore::Tensor &input) const {
    auto local = ColumnParallelLinear::forward(input);
    if (tp_size_ == 1) {
        return local;
    }
    const auto rows = local->numel() / local->size(local->ndim() - 1);
    auto gathered = infinicore::op::distributed::allgather(local->view({1, rows, out_features_ / tp_size_}), tp_size_, communicator_);
    auto shape = local->shape();
    shape.back() = out_features_;
    return gathered->permute({1, 0, 2})->contiguous()->view(shape);
}

infinicore::Tensor VocabParallelLinear::top_tokens(infinicore::Tensor &input) const {
    auto logits = ColumnParallelLinear::forward(input);
    const auto vocab = logits->size(logits->ndim() - 1);
    const auto rows = logits->numel() / vocab;
    logits = logits->view({rows, vocab});
    auto ids = infinicore::Tensor::empty({rows}, infinicore::DataType::I64, input->device());
    auto scores = tp_size_ > 1
                    ? infinicore::Tensor::empty({rows}, logits->dtype(), input->device())
                    : infinicore::Tensor{};
    for (size_t i = 0; i < rows; ++i) {
        auto row = logits->narrow({{0, i, 1}})->view({vocab});
        auto id = ids->narrow({{0, i, 1}});
        // The existing greedy sampler chooses the lowest index on ties.
        infinicore::op::random_sample_(id->view({}), row, 0.0f, 1.0f, 1, 1.0f);
        if (tp_size_ > 1) {
            infinicore::op::take_(scores->narrow({{0, i, 1}}), row, id);
        }
    }
    if (tp_size_ == 1) {
        return ids;
    }
    ids = infinicore::op::add(ids, vocab_start_->as_strided({rows}, {0})->contiguous());
    auto rank_ids = infinicore::op::distributed::allgather(ids->view({1, rows}), tp_size_, communicator_)->permute({1, 0})->contiguous();
    auto rank_scores = infinicore::op::distributed::allgather(scores->view({1, rows}), tp_size_, communicator_)->permute({1, 0})->contiguous();
    auto output = infinicore::Tensor::empty({rows}, infinicore::DataType::I64, input->device());
    auto winner = infinicore::Tensor::empty({1}, infinicore::DataType::I64, input->device());
    for (size_t i = 0; i < rows; ++i) {
        auto row = rank_scores->narrow({{0, i, 1}})->view({tp_size_});
        infinicore::op::random_sample_(winner->view({}), row, 0.0f, 1.0f, 1, 1.0f);
        // Rank order matches increasing global token ID, preserving tie rules.
        infinicore::op::take_(output->narrow({{0, i, 1}}), rank_ids->narrow({{0, i, 1}}), winner);
    }
    return output;
}

} // namespace infinilm::nn
