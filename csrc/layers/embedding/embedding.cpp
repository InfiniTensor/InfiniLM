#include "embedding.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "../../execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "base/embedding.h"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::layers {
using infinilm::runtime::DataType;
using infinilm::runtime::Device;
using infinilm::runtime::Runtime;
using infinilm::runtime::Shape;
using infinilm::runtime::Size;
using infinilm::runtime::Tensor;

Embedding::Embedding(Runtime &runtime, size_t num_embeddings, size_t embedding_dim,
                     std::optional<int64_t> padding_idx, const DataType &dtype,
                     size_t tp_rank, size_t tp_size, infinicclComm_t communicator)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
      padding_idx_(padding_idx), dtype_(dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Embedding construction requires eager Runtime");
    }
    if (!num_embeddings || !embedding_dim
        || (padding_idx && (*padding_idx < 0 || static_cast<uint64_t>(*padding_idx) >= num_embeddings))) {
        throw std::invalid_argument("Invalid explicit Embedding construction geometry");
    }
    device_ = runtime.device();
    if (!tp_size || tp_rank >= tp_size || embedding_dim % tp_size || (tp_size > 1 && !communicator)) {
        throw std::invalid_argument("Embedding TP geometry or communicator is invalid");
    }
    tp_rank_ = tp_rank;
    tp_size_ = tp_size;
    communicator_ = communicator;
    try {
        weight_ = infinilm::weights::Weight(runtime, {num_embeddings, embedding_dim}, dtype_, tp_size_ > 1 ? 1 : 0, tp_rank_, tp_size_);
        this->bind_weight("weight", weight_);
        // Same legacy semantics: padding row is not additionally zero-initialized.
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

Tensor Embedding::forward(Runtime &runtime, const Tensor &indices) const {
    runtime.assertOwnerThread();
    const auto weights = weight_.tensor();
    if (runtime.device() != device_ || indices->device() != device_ || weights->device() != device_) {
        throw std::invalid_argument("Embedding Runtime, indices and weight devices must agree");
    }
    auto output_shape = indices->shape();
    output_shape.push_back(embedding_dim_);
    auto out = Tensor::empty(runtime, output_shape, weights->dtype());
    auto input = indices->contiguous(runtime);
    const auto config = infini::ops::Embedding::DefaultConfig(
        runtime.device().type());
    auto local_out = out;
    if (tp_size_ > 1) {
        infinilm::execution::submit<infini::ops::Fill>(runtime,
                                                       infini::ops::Fill::DefaultConfig(runtime.device().type()), out, 0.0, out);
        local_out = out->narrow({{out->ndim() - 1, tp_rank_ * (embedding_dim_ / tp_size_), embedding_dim_ / tp_size_}});
    }
    infinilm::execution::submit<infini::ops::Embedding>(runtime, config, input, weights, local_out);
    if (tp_size_ > 1) {
        infinilm::engine::distributed::allreduce_(runtime, out, out, infinicclSum, communicator_);
        // A first-use kernel module can synchronize a device while holding a
        // CUDA runtime lock needed by the peer to enqueue this collective.
        // Complete the new eager collective before dispatching decoder kernels.
        if (!runtime.isGraphRecording()) {
            runtime.syncStream();
        }
    }
    if (device_.type() != Device::Type::kCpu) {
        INFINILM_TRACE_OP(Embedding, out);
    }
    return out;
}

std::string Embedding::extra_repr() const {
    std::string repr = "Embedding(num_embeddings=" + std::to_string(num_embeddings_) + ", embedding_dim=" + std::to_string(embedding_dim_) + ", dtype=" + std::to_string(static_cast<int>(dtype_));
    if (padding_idx_.has_value()) {
        repr += ", padding_idx=" + std::to_string(padding_idx_.value());
    }
    repr += ")";
    return repr;
}

} // namespace infinilm::layers
