#pragma once

#include "../ops.hpp"
#include "../quantization.hpp"
#include "module.hpp"
#include <infiniccl.h>
#include <optional>

namespace infinicore::nn {

class BaseLinear : public Module {
public:
    BaseLinear(size_t in_features, size_t out_features, bool bias = true,
               const DataType &dtype = DataType::F32, const Device &device = Device());

    BaseLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias = true,
               const DataType &dtype = DataType::F32, const Device &device = Device());

    // Forward pass: output = input @ weight.T + bias
    Tensor forward(Tensor &input) const;

    // Forward pass with residual connection (InfiniLM-style)
    // output = input @ weight.T + bias + residual
    Tensor forward(Tensor &input, Tensor &residual) const;

    // Module information
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    bool has_bias() const { return has_bias_; }
    DataType dtype() const { return dtype_; }

    // Accessors for parameters
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }
    Tensor weight_scale() const { return weight_scale_; }
    Tensor weight_zeros() const { return weight_zeros_; }
    Tensor gidx() const { return gidx_; }

    std::shared_ptr<infinicore::quantization::BaseQuantization> get_quantization() const { return quantization_; }

protected:
    // Parameters
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);

    INFINICORE_NN_PARAMETER(weight_scale);
    INFINICORE_NN_PARAMETER(weight_zeros);

    INFINICORE_NN_PARAMETER(gidx);

protected:
    // Helper method for common forward computation
    Tensor compute_linear(Tensor &input) const;

    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    DataType dtype_;
    std::shared_ptr<infinicore::quantization::BaseQuantization> quantization_ = std::make_shared<infinicore::quantization::NoneQuantization>(nullptr);
};

} // namespace infinicore::nn

namespace infinicore::nn {

class Linear : public BaseLinear {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true,
           const DataType &dtype = DataType::F32, const Device &device = Device());

    Linear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias = true,
           const DataType &dtype = DataType::F32, const Device &device = Device());

    // Forward pass: output = input @ weight.T + bias
    Tensor forward(Tensor &input) const;

    // String representation
    std::string extra_repr() const;
};

class ColumnParallelLinear : public BaseLinear {
public:
    ColumnParallelLinear(size_t in_features, size_t out_features, bool bias = true,
                         const DataType &dtype = DataType::F32, const Device &device = Device(),
                         Size tp_rank = 0, Size tp_size = 1);

    ColumnParallelLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias = true,
                         const DataType &dtype = DataType::F32, const Device &device = Device(),
                         Size tp_rank = 0, Size tp_size = 1);

    // Forward pass: output = input @ weight.T + bias
    Tensor forward(Tensor &input) const;

    // String representation
    std::string extra_repr() const;

protected:
    Size tp_rank_ = 0;
    Size tp_size_ = 1;
};

class RowParallelLinear : public BaseLinear {
public:
    RowParallelLinear(size_t in_features, size_t out_features, bool bias = true,
                      const DataType &dtype = DataType::F32, const Device &device = Device(),
                      Size tp_rank = 0, Size tp_size = 1, infinicclComm_t communicator = nullptr);

    RowParallelLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias = true,
                      const DataType &dtype = DataType::F32, const Device &device = Device(),
                      Size tp_rank = 0, Size tp_size = 1, infinicclComm_t communicator = nullptr);

    // Forward pass: output = input @ weight.T + bias
    Tensor forward(Tensor &input) const;

    // String representation
    std::string extra_repr() const;

protected:
    Size tp_rank_ = 0;
    Size tp_size_ = 1;
    infinicclComm_t communicator_;
};

} // namespace infinicore::nn
