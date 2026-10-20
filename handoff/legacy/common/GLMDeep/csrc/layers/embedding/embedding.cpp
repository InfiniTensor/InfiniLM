#include "../../weights/weights.hpp"
#include "embedding.hpp"
#include "../../execution/submit.hpp"
#include "base/embedding.h"
#include <exception>
#include "runtime/runtime.hpp"
#include <stdexcept>

namespace infinilm::layers {
using infinilm::runtime::Tensor;
using infinilm::runtime::Runtime;
using infinilm::runtime::DataType;
using infinilm::runtime::Device;
using infinilm::runtime::Shape;
using infinilm::runtime::Size;


Embedding::Embedding(Runtime &runtime, size_t num_embeddings, size_t embedding_dim,
                     std::optional<int64_t> padding_idx, const DataType &dtype)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
      padding_idx_(padding_idx), dtype_(dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Embedding construction requires eager Runtime");
    if (!num_embeddings || !embedding_dim
        || (padding_idx && (*padding_idx < 0 || static_cast<uint64_t>(*padding_idx) >= num_embeddings)))
        throw std::invalid_argument("Invalid explicit Embedding construction geometry");
    device_ = runtime.device();
    try {
        weight_ = infinilm::weights::Weight(runtime, {num_embeddings, embedding_dim}, dtype_);
        this->bind_weight("weight", weight_);
        // Same legacy semantics: padding row is not additionally zero-initialized.
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
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
    infinilm::execution::submit<infini::ops::Embedding>(runtime, config, input, weights, out);
    if (device_.type() != Device::Type::kCpu) INFINILM_TRACE_OP(Embedding, out);
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
