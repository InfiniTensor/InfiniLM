#include "linear.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/matmul_allreduce_add_rmsnorm_ascend.hpp"
#include "infinicore/ops/matmul_allreduce_ascend.hpp"
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

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
    static const bool vendor_enabled = []() {
        const char *value = std::getenv("INFINILM_ASCEND_MATMUL_ALLREDUCE_VENDOR");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    const bool can_use_vendor = vendor_enabled
                             && input->device().getType() == infinicore::Device::Type::ASCEND
                             && tp_size_ > 1
                             && communicator_ != nullptr
                             && !has_bias_
                             && alpha_ == 1.0f
                             && quantization_->get_quant_scheme()
                                    == infinilm::quantization::QuantScheme::NONE;

    if (can_use_vendor) {
        auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
        auto weight = static_cast<const infinicore::Tensor &>(
            parameters_.at("weight"));
        auto weight_transposed = weight_transposed_view_;
        if (!weight_transposed) {
            weight_transposed = weight->contiguous()->permute({1, 0});
        }
        size_t rows = 1;
        for (size_t i = 0; i + 1 < input_contiguous->ndim(); ++i) {
            rows *= input_contiguous->shape()[i];
        }
        auto output_2d = infinicore::op::matmul_allreduce_ascend(
            input_contiguous->view({rows, weight_transposed->shape()[0]}),
            weight_transposed,
            communicator_);
        auto output_shape = input_contiguous->shape();
        output_shape.back() = weight_transposed->shape()[1];
        return output_2d->view(output_shape);
    }

    auto output = BaseLinear::forward(input);

    if ((tp_size_ > 1) && (communicator_ != nullptr)) {
        infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
    }
    return output;
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
RowParallelLinear::forward_add_rmsnorm(
    infinicore::Tensor &input,
    const infinicore::Tensor &residual,
    const infinicore::Tensor &gamma,
    float epsilon) const {
    static const bool vendor_enabled = []() {
        const char *value = std::getenv(
            "INFINILM_ASCEND_MC2_ADD_RMSNORM_VENDOR");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    const bool can_use_vendor = vendor_enabled
                             && input->device().getType()
                                    == infinicore::Device::Type::ASCEND
                             && tp_size_ > 1
                             && communicator_ != nullptr
                             && !has_bias_
                             && alpha_ == 1.0f
                             && quantization_->get_quant_scheme()
                                    == infinilm::quantization::QuantScheme::NONE
                             && input->dtype() == residual->dtype()
                             && input->dtype() == gamma->dtype();

    if (can_use_vendor) {
        auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
        auto residual_contiguous = residual->is_contiguous()
                                     ? residual
                                     : residual->contiguous();
        auto gamma_contiguous = gamma->is_contiguous() ? gamma : gamma->contiguous();
        auto weight = static_cast<const infinicore::Tensor &>(
            parameters_.at("weight"));
        auto weight_contiguous = weight->is_contiguous() ? weight : weight->contiguous();

        size_t rows = 1;
        for (size_t i = 0;
             i + 1 < input_contiguous->ndim();
             ++i) {
            rows *= input_contiguous->shape()[i];
        }
        auto input_2d = input_contiguous->view(
            {rows, input_contiguous->shape().back()});
        auto residual_2d = residual_contiguous->view(
            {rows, residual_contiguous->shape().back()});
        auto [normalized_2d, add_out_2d] = infinicore::op::
            matmul_allreduce_add_rmsnorm_ascend(
                input_2d,
                weight_contiguous,
                residual_2d,
                gamma_contiguous,
                communicator_,
                epsilon);
        return {
            normalized_2d->view(residual->shape()),
            add_out_2d->view(residual->shape())};
    }

    auto projected = forward(input);
    auto add_out = infinicore::op::add(residual, projected);
    auto normalized = infinicore::op::rms_norm(add_out, gamma, epsilon);
    return {normalized, add_out};
}

std::string RowParallelLinear::extra_repr() const {
    return "RowParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::nn
