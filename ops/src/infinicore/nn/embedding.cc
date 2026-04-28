#include "infinicore/nn/embedding.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinicore::nn {

Embedding::Embedding(size_t num_embeddings,
                     size_t embedding_dim,
                     std::optional<int64_t> padding_idx,
                     const DataType &dtype,
                     const Device &device)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      padding_idx_(padding_idx),
      dtype_(dtype) {

    device_ = device;

    // Validate padding_idx
    if (padding_idx_.has_value()) {
        int64_t idx = padding_idx_.value();
        if (idx < 0 || idx >= static_cast<int64_t>(num_embeddings)) {
            throw std::invalid_argument(
                "padding_idx must be within num_embeddings range, got " + std::to_string(idx) + " for num_embeddings=" + std::to_string(num_embeddings));
        }
    }

    // Initialize parameter using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_embeddings, embedding_dim}, dtype_, device));

    // If padding_idx is specified, initialize that row to zeros
    if (padding_idx_.has_value()) {
        // TODO: Set weight[padding_idx] to zeros
        // This would require a slice operation
    }

    SPDLOG_DEBUG("Created Embedding module: num_embeddings={}, embedding_dim={}, dtype={}, padding_idx={}",
                 num_embeddings, embedding_dim, static_cast<int>(dtype_),
                 padding_idx_.has_value() ? std::to_string(padding_idx_.value()) : "None");
}

Tensor Embedding::forward(const Tensor &indices) const {
    // TODO: Implement on-device embedding for all devices, then remove the condition and the classic approach
    auto device_type = device_.getType();
    if (device_type == Device::Type::NVIDIA || device_type == Device::Type::ILUVATAR || device_type == Device::Type::METAX || device_type == Device::Type::MOORE || device_type == Device::Type::ALI || device_type == Device::Type::QY || device_type == Device::Type::HYGON) {
        // Use op::embedding which supports device-side input and batch dimension
        return op::embedding(indices->contiguous()->to(device_), weight_);
    }

    // Get the shape of indices
    auto indices_shape = indices->shape();

    // Output shape: indices_shape + [embedding_dim]
    std::vector<size_t> output_shape = indices_shape;
    output_shape.push_back(embedding_dim_);

    // Create output tensor on the same device as weight
    auto out = Tensor::empty(output_shape, weight_->dtype(), weight_->device());

    // Flatten indices for sequential row copies
    auto cpu_device = Device(Device::Type::CPU, 0);
    auto indices_cpu = indices->to(cpu_device)->contiguous();

    // Calculate total number of lookups
    size_t num_lookups = 1;
    for (auto dim : indices_shape) {
        num_lookups *= dim;
    }

    const size_t row_bytes = embedding_dim_ * dsize(weight_->dtype());

    // Source and destination base pointers
    auto *weight_base = weight_->data();
    auto *out_base = out->data();

    // Helper lambda to read index based on dtype with bounds checking
    auto read_index = [&](size_t i) -> int64_t {
        auto dtype = indices_cpu->dtype();
        if (dtype == DataType::I32) {
            const auto *data = reinterpret_cast<const int32_t *>(indices_cpu->data());
            return static_cast<int64_t>(data[i]);
        } else if (dtype == DataType::I64) {
            const auto *data = reinterpret_cast<const int64_t *>(indices_cpu->data());
            return data[i];
        } else if (dtype == DataType::U32) {
            const auto *data = reinterpret_cast<const uint32_t *>(indices_cpu->data());
            return static_cast<int64_t>(data[i]);
        } else if (dtype == DataType::U64) {
            const auto *data = reinterpret_cast<const uint64_t *>(indices_cpu->data());
            uint64_t val = data[i];
            // Check if value can fit in int64_t
            if (val > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::out_of_range("Index value out of range for int64_t: " + std::to_string(val));
            }
            return static_cast<int64_t>(val);
        } else {
            throw std::runtime_error("Embedding indices must be integer type, got dtype=" + std::to_string(static_cast<int>(dtype)));
        }
    };

    if (weight_->device().getType() == Device::Type::CPU) {
        // CPU path: memcpy row by row
        for (size_t i = 0; i < num_lookups; ++i) {
            int64_t idx = read_index(i);
            if (idx < 0 || idx >= static_cast<int64_t>(num_embeddings_)) {
                throw std::out_of_range(
                    "Index out of range: " + std::to_string(idx) + " (num_embeddings=" + std::to_string(num_embeddings_) + ")");
            }
            std::memcpy(out_base + i * row_bytes, weight_base + idx * row_bytes, row_bytes);
        }
    } else {
        // Device path: use stream-ordered D2D copies
        for (size_t i = 0; i < num_lookups; ++i) {
            int64_t idx = read_index(i);
            if (idx < 0 || idx >= static_cast<int64_t>(num_embeddings_)) {
                throw std::out_of_range(
                    "Index out of range: " + std::to_string(idx) + " (num_embeddings=" + std::to_string(num_embeddings_) + ")");
            }
            context::memcpyD2D(out_base + i * row_bytes, weight_base + idx * row_bytes, row_bytes);
        }
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

} // namespace infinicore::nn
