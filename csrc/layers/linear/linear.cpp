#include "linear.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include <optional>

namespace infinilm::nn {

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

infinicore::Tensor RowParallelLinear::forward(infinicore::Tensor &input) const {
    auto output = BaseLinear::forward(input);

    if (needs_allreduce()) {
        auto &ctx = infinilm::global_state::get_forward_context();
        if (ctx.defer_row_parallel_allreduce) {
            ctx.deferred_allreduces.push_back({output, communicator_});
        } else {
            allreduce_output(output);
        }
    }
    return output;
}

infinicore::Tensor RowParallelLinear::forward_matmul_only(infinicore::Tensor &input) const {
    return BaseLinear::forward(input);
}

void RowParallelLinear::allreduce_output(infinicore::Tensor &output) const {
    if (needs_allreduce()) {
        if (output->is_contiguous()) {
            infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
        } else {
            auto contiguous = output->contiguous();
            infinicore::op::distributed::allreduce_(contiguous, contiguous, INFINICCL_SUM, communicator_);
            output->copy_from(contiguous);
        }
    }
}

bool RowParallelLinear::needs_allreduce() const {
    return (tp_size_ > 1) && (communicator_ != nullptr);
}

std::string RowParallelLinear::extra_repr() const {
    return "RowParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::nn
