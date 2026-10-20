#include "dense.hpp"
#include "linear.hpp"
#include "runtime/runtime.hpp"
#include <stdexcept>

namespace infinilm::nn {

namespace {
infinilm::runtime::Runtime &checked_row_runtime(infinilm::runtime::Runtime &runtime, infinilm::runtime::Size tp_size,
                                        infinicclComm_t communicator) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Row Linear construction requires eager Runtime");
    if (tp_size > 1 && !communicator) throw std::invalid_argument("Row Linear requires its TP communicator");
    return runtime;
}
} // namespace
Linear::Linear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features, bool bias,
               const infinilm::runtime::DataType &dtype,
               std::optional<infinilm::runtime::DataType> output_dtype)
    : BaseLinear(runtime, in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr, output_dtype),
                 bias, dtype, -1, 0, 1) {
}

Linear::Linear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features,
               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
               bool bias, const infinilm::runtime::DataType &dtype)
    : BaseLinear(runtime, in_features, out_features, quantization, bias, dtype, -1, 0, 1) {
}

ColumnParallelLinear::ColumnParallelLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features, bool bias,
                                           const infinilm::runtime::DataType &dtype,
                                           infinilm::runtime::Size tp_rank, infinilm::runtime::Size tp_size,
                                           int tp_num_heads)
    : BaseLinear(runtime, in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, 0, tp_rank, tp_size, tp_num_heads),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {
}

ColumnParallelLinear::ColumnParallelLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           bool bias, const infinilm::runtime::DataType &dtype,
                                           infinilm::runtime::Size tp_rank, infinilm::runtime::Size tp_size,
                                           int tp_num_heads)
    : BaseLinear(runtime, in_features, out_features, quantization, bias, dtype,
                 0, tp_rank, tp_size, tp_num_heads),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {
}

RowParallelLinear::RowParallelLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features, bool bias,
                                     const infinilm::runtime::DataType &dtype,
                                     infinilm::runtime::Size tp_rank, infinilm::runtime::Size tp_size,
                                     infinicclComm_t communicator)
    : BaseLinear(checked_row_runtime(runtime, tp_size, communicator), in_features, out_features,
                 std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
                 bias, dtype, 1, tp_rank, tp_size),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {
}

RowParallelLinear::RowParallelLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias, const infinilm::runtime::DataType &dtype,
                                     infinilm::runtime::Size tp_rank, infinilm::runtime::Size tp_size,
                                     infinicclComm_t communicator)
    : BaseLinear(checked_row_runtime(runtime, tp_size, communicator), in_features, out_features, quantization, bias, dtype,
                 1, tp_rank, tp_size),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {
}

// ---- RowParallelLinear ----

infinilm::runtime::Tensor RowParallelLinear::forward(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) const {
    if (tp_size_ > 1 && communicator_ != nullptr) {
        return compute_linear_allreduce(runtime, input, communicator_);
}
    return BaseLinear::forward(runtime, input);
}

} // namespace infinilm::nn

namespace infinilm::layers::linear {

std::optional<infinilm::runtime::DataType> prepare_lm_head_output_dtype(
    const infinilm::runtime::Device &device, infinilm::runtime::DataType weight_dtype,
    std::optional<infinilm::runtime::DataType> requested_output_dtype) {
    if (!requested_output_dtype.has_value()) {
        return std::nullopt;
    }
    if (*requested_output_dtype != infinilm::runtime::DataType::kFloat32
        || weight_dtype != infinilm::runtime::DataType::kBFloat16) {
        throw std::invalid_argument(
            "lm_head_output_dtype requires float32 output and bfloat16 weights");
    }
    // Share the literal Linear contract; this does not probe or choose a provider.
    (void)infinilm::layers::linear::resolve_output_dtype(
        device, weight_dtype, weight_dtype, false, requested_output_dtype);
    return requested_output_dtype;
}

} // namespace infinilm::layers::linear
