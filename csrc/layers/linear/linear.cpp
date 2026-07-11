#include "linear.hpp"
#include "../../engine/distributed/async_collective.hpp"
#include "../../engine/workspace/workspace_context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include <infinicore/context/context.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::nn {
namespace {

std::optional<size_t> parse_chunk_rows_env() {
    const char *raw = std::getenv("INFINILM_COMM_OVERLAP_CHUNK_ROWS");
    if (raw == nullptr || raw[0] == '\0') {
        return std::nullopt;
    }
    char *end = nullptr;
    const auto value = std::strtoull(raw, &end, 10);
    if (end == raw) {
        return std::nullopt;
    }
    return static_cast<size_t>(value);
}

size_t default_comm_overlap_chunk_rows(infinicore::Size tp_size) {
    if (tp_size > 2 && infinilm::engine::distributed::async_collectives_force_enabled_by_env()) {
        return 2048;
    }
    return 8192;
}

size_t comm_overlap_chunk_rows(size_t rows, infinicore::Size tp_size) {
    constexpr size_t kMaxPendingChunks = 32;
    const size_t requested = parse_chunk_rows_env().value_or(default_comm_overlap_chunk_rows(tp_size));
    if (requested == 0) {
        return 0;
    }
    const size_t min_chunk = (rows + kMaxPendingChunks - 1) / kMaxPendingChunks;
    return std::max(requested, min_chunk);
}

bool can_use_async_comm_overlap(const infinicore::Tensor &input, infinicore::Size tp_size) {
    return infinilm::engine::distributed::async_collectives_supported_by_policy(input->device(), static_cast<int>(tp_size));
}

} // namespace

// ---- Linear ----

Linear::Linear(size_t in_features, size_t out_features, bool bias,
               const infinicore::DataType &dtype, const infinicore::Device &device)
    : BaseLinear(in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, device, -1, 0, 1) {
}

Linear::Linear(size_t in_features, size_t out_features,
               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
               bool bias, const infinicore::DataType &dtype, const infinicore::Device &device)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device, -1, 0, 1) {
}

infinicore::Tensor Linear::forward(infinicore::Tensor &input) const {
    return BaseLinear::forward(input);
}

std::string Linear::extra_repr() const {
    return "Linear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

// ---- ColumnParallelLinear ----

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           infinicore::Size tp_rank, infinicore::Size tp_size,
                                           int tp_num_heads)
    : BaseLinear(in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, device, 0, tp_rank, tp_size, tp_num_heads),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {
}

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           bool bias, const infinicore::DataType &dtype, const infinicore::Device &device,
                                           infinicore::Size tp_rank, infinicore::Size tp_size,
                                           int tp_num_heads)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device,
                 0, tp_rank, tp_size, tp_num_heads),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {
}

infinicore::Tensor ColumnParallelLinear::forward(infinicore::Tensor &input) const {
    return BaseLinear::forward(input);
}

std::string ColumnParallelLinear::extra_repr() const {
    return "ColumnParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

// ---- RowParallelLinear ----

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features, bool bias,
                                     const infinicore::DataType &dtype, const infinicore::Device &device,
                                     infinicore::Size tp_rank, infinicore::Size tp_size,
                                     infinicclComm_t communicator)
    : BaseLinear(in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, device, 1, tp_rank, tp_size),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {
}

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias, const infinicore::DataType &dtype, const infinicore::Device &device,
                                     infinicore::Size tp_rank, infinicore::Size tp_size,
                                     infinicclComm_t communicator)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device,
                 1, tp_rank, tp_size),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {
}

bool RowParallelLinearOutput::has_pending() const {
    return async_context != nullptr && !pending_collectives.empty();
}

void RowParallelLinearOutput::wait() const {
    if (async_context == nullptr) {
        return;
    }
    for (const auto &pending : pending_collectives) {
        async_context->wait(pending);
    }
}

infinicore::Tensor RowParallelLinear::forward(infinicore::Tensor &input) const {
    auto result = forward_async(input);
    result.wait();
    return result.output;
}

RowParallelLinearOutput RowParallelLinear::forward_async(infinicore::Tensor &input) const {
    const bool needs_allreduce = (tp_size_ > 1) && (communicator_ != nullptr);
    auto *workspace = infinilm::engine::maybe_current_workspace();
    auto *async_context = infinilm::engine::distributed::maybe_current_async_collective_context();
    const bool can_async_allreduce = needs_allreduce && async_context != nullptr && async_context->enabled() && !infinicore::context::isGraphRecording() && can_use_async_comm_overlap(input, tp_size_);
    const bool supports_preallocated_output = quantization_->supports_preallocated_output();

    auto output_shape = input->shape();
    output_shape.back() = out_features_;
    const auto input_shape = input->shape();
    const size_t input_features = input_shape.back();
    const size_t rows = input->numel() / input_features;
    const size_t chunk_rows = comm_overlap_chunk_rows(rows, tp_size_);
    const bool use_chunked_overlap = can_async_allreduce && supports_preallocated_output && chunk_rows > 0 && rows > chunk_rows;

    if (use_chunked_overlap) {
        infinicore::Tensor partial_output;
        infinicore::Tensor output;
        if (workspace != nullptr) {
            partial_output = workspace->acquire_temp_tensor(
                "row_parallel.chunked_allreduce_input",
                output_shape,
                input->dtype(),
                256);
            output = workspace->acquire_temp_tensor(
                "row_parallel.chunked_allreduce_output",
                output_shape,
                input->dtype(),
                256);
        } else {
            partial_output = infinicore::Tensor::empty(output_shape, input->dtype(), input->device());
            output = infinicore::Tensor::empty(output_shape, input->dtype(), input->device());
        }

        auto input_flat = input->view({rows, input_features});
        auto partial_flat = partial_output->view({rows, out_features_});
        auto output_flat = output->view({rows, out_features_});

        std::vector<infinilm::engine::distributed::PendingCollective> pending_collectives;
        pending_collectives.reserve((rows + chunk_rows - 1) / chunk_rows);
        for (size_t begin = 0; begin < rows; begin += chunk_rows) {
            const size_t len = std::min(chunk_rows, rows - begin);
            auto input_chunk = input_flat->narrow({{0, begin, len}});
            auto partial_chunk = partial_flat->narrow({{0, begin, len}});
            auto output_chunk = output_flat->narrow({{0, begin, len}});
            compute_linear(input_chunk, &partial_chunk);
            pending_collectives.push_back(async_context->allreduce(
                output_chunk,
                partial_chunk,
                INFINICCL_SUM,
                communicator_));
        }
        return RowParallelLinearOutput{output, partial_output, async_context, std::move(pending_collectives)};
    }

    const bool use_workspace_output = needs_allreduce && workspace != nullptr && supports_preallocated_output;

    infinicore::Tensor output;
    infinicore::Tensor allreduce_input;
    std::optional<infinicore::Tensor> preallocated_partial_output;
    std::optional<infinicore::Tensor> preallocated_reduced_output;
    if (use_workspace_output) {
        preallocated_partial_output.emplace(workspace->acquire_temp_tensor(
            "row_parallel.allreduce_input",
            output_shape,
            input->dtype(),
            256));
        allreduce_input = compute_linear(input, &preallocated_partial_output.value());
        preallocated_reduced_output.emplace(workspace->acquire_temp_tensor(
            "row_parallel.allreduce_output",
            output_shape,
            input->dtype(),
            256));
        output = preallocated_reduced_output.value();

        if (infinicore::context::isGraphRecording()) {
            const auto key = workspace->next_collective_key("row_parallel.allreduce");
            const auto status = infinicclCommRegisterAllReduceBuffer(
                communicator_,
                key.c_str(),
                const_cast<std::byte *>(allreduce_input->data()),
                allreduce_input->nbytes());
            if (status != INFINI_STATUS_SUCCESS) {
                throw std::runtime_error("failed to register row-parallel allreduce workspace buffer: " + std::to_string(static_cast<int>(status)));
            }
        }
    } else {
        output = BaseLinear::forward(input);
        allreduce_input = output;
    }

    std::vector<infinilm::engine::distributed::PendingCollective> pending_collectives;
    if (needs_allreduce) {
        if (can_async_allreduce) {
            pending_collectives.push_back(async_context->allreduce(output, allreduce_input, INFINICCL_SUM, communicator_));
        } else {
            infinicore::op::distributed::allreduce_(output, allreduce_input, INFINICCL_SUM, communicator_);
        }
    }
    return RowParallelLinearOutput{output, allreduce_input, can_async_allreduce ? async_context : nullptr, std::move(pending_collectives)};
}

std::string RowParallelLinear::extra_repr() const {
    return "RowParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::nn
