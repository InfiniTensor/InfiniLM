#include "infinicore/nn/linear.hpp"
#include "../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/linear_w4a16_awq.hpp"
#include "infinicore/ops/linear_w4a16_gptq_qy.hpp"
#include "infinicore/ops/linear_w8a8i8.hpp"
#include <optional>
#include <spdlog/spdlog.h>

namespace infinicore::nn {

BaseLinear::BaseLinear(size_t in_features, size_t out_features, bool bias,
                       const DataType &dtype, const Device &device)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias),
      dtype_(dtype) {

    device_ = device;
}

BaseLinear::BaseLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
                       const DataType &dtype, const Device &device)
    : in_features_(in_features),
      out_features_(out_features),
      quantization_(quantization),
      has_bias_(bias),
      dtype_(dtype) {

    device_ = device;
}

Tensor BaseLinear::compute_linear(Tensor &input) const {
    switch (this->quantization_->get_quant_scheme()) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

        Tensor weight_packed_tensor = static_cast<const Tensor &>(weight_);
        Tensor weight_scale_tensor = static_cast<const Tensor &>(weight_scale_);
        // weight_packed should be transposed and non-contiguous.
        std::optional<Tensor> bias_opt = has_bias_ ? std::make_optional<Tensor>(static_cast<const Tensor &>(bias_)) : std::nullopt;

        auto output = infinicore::op::linear_w8a8i8(input_contiguous->contiguous(), weight_packed_tensor, weight_scale_tensor, bias_opt);
        return output;
    }
    case infinicore::quantization::QuantScheme::AWQ_W4A16: {
        Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();
        Tensor qweight = static_cast<const Tensor &>(weight_);
        Tensor qzeros = static_cast<const Tensor &>(weight_zeros_);
        Tensor scales = static_cast<const Tensor &>(weight_scale_);
        std::optional<Tensor> bias_opt = has_bias_ ? std::make_optional<Tensor>(static_cast<const Tensor &>(bias_)) : std::nullopt;
        auto output = infinicore::op::linear_w4a16_awq(input_contiguous->contiguous(), qweight, scales, qzeros, bias_opt);
        return output;
    }
    case infinicore::quantization::QuantScheme::GPTQ_W4A16_QY: {
        Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();
        Tensor qweight = static_cast<const Tensor &>(weight_);
        Tensor qzeros = static_cast<const Tensor &>(weight_zeros_);
        Tensor scales = static_cast<const Tensor &>(weight_scale_);
        Tensor g_idx = static_cast<const Tensor &>(gidx_);
        std::optional<Tensor> bias_opt = has_bias_ ? std::make_optional<Tensor>(static_cast<const Tensor &>(bias_)) : std::nullopt;
        auto output = infinicore::op::linear_w4a16_gptq_qy(input_contiguous->contiguous(), qweight, qzeros, scales, 0, 4);
        if (bias_opt.has_value()) {
            infinicore::op::add_(output, output, bias_opt.value()->as_strided(output->shape(), {0, 0, 1}));
        }
        return output;
    }
    default: {
        // Ensure input is contiguous before creating views (required for matmul)
        // This prevents hanging when input tensor has non-contiguous memory layout
        Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

        // Use ops::linear_ directly to match Python backend's exact code path
        // This ensures identical computation and numerical results
        // Parameter inherits from Tensor, so we cast to Tensor explicitly
        Tensor weight_tensor = static_cast<const Tensor &>(weight_);
        std::optional<Tensor> bias_opt = has_bias_ ? std::make_optional<Tensor>(static_cast<const Tensor &>(bias_)) : std::nullopt;

        auto output = infinicore::op::linear(input_contiguous->contiguous(), weight_tensor->contiguous(), bias_opt);
        return output;
    }
    }
} // namespace infinicore::nn

Tensor BaseLinear::forward(Tensor &input) const {
    return compute_linear(input);
}

Tensor BaseLinear::forward(Tensor &input, Tensor &residual) const {
    auto output = compute_linear(input);

    // Add residual: output = output + residual
    infinicore::op::add_(output, output, residual);

    return output;
}

} // namespace infinicore::nn

namespace infinicore::nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias,
               const DataType &dtype, const Device &device)
    : BaseLinear(in_features, out_features, bias, dtype, device_) {

    device_ = device;

    // Initialize parameters using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device));

    // Register bias parameter if requested
    if (bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
    } else {
        bias_ = Parameter(); // Default constructed empty parameter
    }

    // SPDLOG_DEBUG("Created Linear module: in_features={}, out_features={}, bias={}, dtype={}",
    //              in_features, out_features, bias, static_cast<int>(dtype_));
}

Linear::Linear(size_t in_features, size_t out_features,
               std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
               const DataType &dtype, const Device &device)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    case infinicore::quantization::QuantScheme::AWQ_W4A16: {
        weight_ = infinicore::nn::Parameter({out_features, in_features}, infinicore::DataType::I32, device);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({out_features, in_features}, infinicore::DataType::I32, device);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({out_features, in_features}, dtype_, device);
        this->register_parameter("scales", weight_scale_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    case infinicore::quantization::QuantScheme::GPTQ_W4A16_QY: {
        weight_ = infinicore::nn::Parameter({in_features / 2, out_features}, infinicore::DataType::U8, device);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / 128, out_features}, dtype_, device);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / 128, out_features}, dtype_, device);
        this->register_parameter("scales", weight_scale_);

        gidx_ = infinicore::nn::Parameter({in_features}, infinicore::DataType::I32, device);
        this->register_parameter("g_idx", gidx_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    default: {
        // Initialize parameters using macro
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device));

        // Register bias parameter if requested
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = Parameter(); // Default constructed empty parameter
        }

        // SPDLOG_DEBUG("Created Linear module: in_features={}, out_features={}, bias={}, dtype={}",
        //              in_features, out_features, bias, static_cast<int>(dtype_));
        break;
    }
    }
}

Tensor Linear::forward(Tensor &input) const {
    return BaseLinear::forward(input);
}

std::string Linear::extra_repr() const {
    return "Linear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn

namespace infinicore::nn {

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features, bool bias,
                                           const DataType &dtype, const Device &device,
                                           Size tp_rank, Size tp_size)
    : BaseLinear(in_features, out_features, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {

    device_ = device;

    // Initialize parameters using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                          0, tp_rank_, tp_size_));

    // Register bias parameter if requested
    if (bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device,
                                            0, tp_rank_, tp_size_));
    } else {
        bias_ = Parameter(); // Default constructed empty parameter
    }
}

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
                                           const DataType &dtype, const Device &device,
                                           Size tp_rank, Size tp_size)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {

        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device, 0, tp_rank_, tp_size_));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device, 0, tp_rank_, tp_size_));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    case infinicore::quantization::QuantScheme::AWQ_W4A16: {
        auto awq_ptr = std::static_pointer_cast<infinicore::quantization::AWQ>(this->quantization_);
        int group_size = awq_ptr->get_group_size();
        int packing_num = awq_ptr->get_packing_num();

        weight_ = infinicore::nn::Parameter({in_features, out_features / packing_num},
                                            infinicore::DataType::I32,
                                            device, 1, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);

        // Weight scale: [out_features, in_features / group_size]
        // One FP32 scale per group of weights (group_size=128)

        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features},
                                                  dtype_,
                                                  device, 1, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);

        // Weight zeros (zero points): [out_features, in_features / group_size]
        // AWQ implementations (e.g., AutoAWQ) typically store zero points as I32
        // for symmetric/asymmetric quantization support
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features / packing_num},
                                                  infinicore::DataType::I32,
                                                  device, 1, tp_rank_, tp_size_);

        this->register_parameter("qzeros", weight_zeros_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    case infinicore::quantization::QuantScheme::GPTQ_W4A16_QY: {
        auto gptq_ptr = std::static_pointer_cast<infinicore::quantization::GPTQ_QY>(this->quantization_);
        int group_size = gptq_ptr->get_group_size();
        int packing_num = gptq_ptr->get_packing_num();
        weight_ = infinicore::nn::Parameter({in_features / 2, out_features}, infinicore::DataType::U8, device, 1, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 1, tp_rank_, tp_size_);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 1, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);
        gidx_ = infinicore::nn::Parameter({in_features},
                                          infinicore::DataType::I32,
                                          device, 0, tp_rank_, tp_size_);
        this->register_parameter("g_idx", gidx_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, tp_rank_, tp_size_));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    default: {
        // Initialize parameters using macro
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                              0, tp_rank_, tp_size_));

        // Register bias parameter if requested
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device,
                                                0, tp_rank_, tp_size_));
        } else {
            bias_ = Parameter(); // Default constructed empty parameter
        }
        break;
    }
    }
}

Tensor ColumnParallelLinear::forward(Tensor &input) const {
    return BaseLinear::forward(input);
}

std::string ColumnParallelLinear::extra_repr() const {
    return "ColumnParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn

namespace infinicore::nn {

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features, bool bias,
                                     const DataType &dtype, const Device &device,
                                     Size tp_rank, Size tp_size, infinicclComm_t communicator)
    : BaseLinear(in_features, out_features, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {

    device_ = device;

    // Initialize parameters using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                          1, tp_rank_, tp_size_));

    // Register bias parameter if requested
    if (bias && (0 == tp_rank_)) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
    } else {
        bias_ = Parameter(); // Default constructed empty parameter
    }
}

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
                                     const DataType &dtype, const Device &device,
                                     Size tp_rank, Size tp_size, infinicclComm_t communicator)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device, 1, tp_rank_, tp_size_));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device, 0, 0, 1));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, tp_rank_, tp_size_));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    case infinicore::quantization::QuantScheme::AWQ_W4A16: {
        // AWQ W4A16 for RowParallelLinear:切分维度为 in_features（权重矩阵的第1维）
        // - Weight: packed int4 in I32 containers (8 int4 per I32)
        // - Group-wise quantization with group_size=128
        // - Scale and zero points stored per group along in_features dimension

        auto awq_ptr = std::static_pointer_cast<infinicore::quantization::AWQ>(this->quantization_);
        int group_size = awq_ptr->get_group_size();
        int packing_num = awq_ptr->get_packing_num();

        // Packed weight: [out_features, in_features / 8]
        weight_ = infinicore::nn::Parameter({in_features, out_features / packing_num},
                                            infinicore::DataType::I32,
                                            device, 0, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);

        // Weight scale: [out_features, in_features / group_size]

        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features},
                                                  dtype_,
                                                  device, 0, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);
        // Weight zeros (zero points): [out_features, in_features / group_size]
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features / packing_num},
                                                  infinicore::DataType::I32,
                                                  device, 0, tp_rank_, tp_size_);
        this->register_parameter("qzeros", weight_zeros_);

        // Bias handling in RowParallelLinear:
        // - Only rank 0 holds the full bias (after all-reduce on output)
        // - Other ranks have empty bias parameter
        if (bias && (0 == tp_rank_)) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    case infinicore::quantization::QuantScheme::GPTQ_W4A16_QY: {
        // GPTQ W4A16 QY for RowParallelLinear:切分维度为 in_features（权重矩阵的第1维）
        // - Weight: packed int4 in U8 containers (8 int4 per U8)
        // - Group-wise quantization with group_size=128
        // - Scale and zero points stored per group along in_features dimension

        auto gptq_ptr = std::static_pointer_cast<infinicore::quantization::GPTQ_QY>(this->quantization_);
        int group_size = gptq_ptr->get_group_size();
        int packing_num = gptq_ptr->get_packing_num();

        weight_ = infinicore::nn::Parameter({in_features / 2, out_features}, infinicore::DataType::U8, device, 0, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 0, tp_rank_, tp_size_);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 0, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);

        gidx_ = infinicore::nn::Parameter({in_features},
                                          infinicore::DataType::I32,
                                          device, 0, tp_rank_, tp_size_);
        this->register_parameter("g_idx", gidx_);
        if (bias && (0 == tp_rank_)) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    default: {
        // Initialize parameters using macro
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                              1, tp_rank_, tp_size_));

        // Register bias parameter if requested
        if (bias && (0 == tp_rank_)) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = Parameter(); // Default constructed empty parameter
        }

        // SPDLOG_DEBUG("Created RowParallelLinear module: in_features={}, out_features={}, bias={}, dtype={}",
        //              in_features, out_features, bias, static_cast<int>(dtype_));
        break;
    }
    }
}

Tensor RowParallelLinear::forward(Tensor &input) const {
    auto output = BaseLinear::forward(input);

    if ((tp_size_ > 1) && (communicator_ != nullptr)) {
        op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
    }
    return output;
}

std::string RowParallelLinear::extra_repr() const {
    return "RowParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
