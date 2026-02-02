#pragma once
#include "infinicore/nn/linear.hpp"
#include "infinicore/quantization.hpp"

#include "../engine/distributed/communication_group.hpp"

namespace infinilm::layers {
class QKVParallelLinear : public infinicore::nn::ColumnParallelLinear {
public:
    explicit QKVParallelLinear(size_t hidden_size,
                               size_t q_dim, size_t k_dim, size_t v_dim,
                               size_t num_q_head, size_t num_k_head, size_t num_v_head,
                               bool q_bias, bool k_bias, bool v_bias,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    // A more common case where all heads have the same dimension
    explicit QKVParallelLinear(size_t hidden_size,
                               size_t head_dim,
                               size_t num_q_head, size_t num_kv_head,
                               bool bias = false,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    explicit QKVParallelLinear(size_t hidden_size,
                               size_t q_dim, size_t k_dim, size_t v_dim,
                               size_t num_q_head, size_t num_k_head, size_t num_v_head,
                               bool q_bias, bool k_bias, bool v_bias,
                               std::shared_ptr<infinicore::quantization::BaseQuantization> quantization,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    // A more common case where all heads have the same dimension
    explicit QKVParallelLinear(size_t hidden_size,
                               size_t head_dim,
                               size_t num_q_head, size_t num_kv_head,
                               std::shared_ptr<infinicore::quantization::BaseQuantization> quantization,
                               bool bias = false,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    forward_split(infinicore::Tensor &input);

    infinicore::nn::Parameter get_q_weight() const;
    infinicore::nn::Parameter get_k_weight() const;
    infinicore::nn::Parameter get_v_weight() const;

    infinicore::nn::Parameter get_q_weight_scale() const;
    infinicore::nn::Parameter get_k_weight_scale() const;
    infinicore::nn::Parameter get_v_weight_scale() const;

    infinicore::nn::Parameter get_q_weight_zeros() const;
    infinicore::nn::Parameter get_k_weight_zeros() const;
    infinicore::nn::Parameter get_v_weight_zeros() const;

    infinicore::nn::Parameter get_q_bias() const;
    infinicore::nn::Parameter get_k_bias() const;
    infinicore::nn::Parameter get_v_bias() const;

    bool has_q_bias() const;
    bool has_k_bias() const;
    bool has_v_bias() const;

private:
    size_t q_dim_;
    size_t k_dim_;
    size_t v_dim_;
    size_t num_q_head_;
    size_t num_k_head_;
    size_t num_v_head_;
    bool q_bias_;
    bool k_bias_;
    bool v_bias_;
    size_t q_out_size_; // num_q_head * q_dim / tp_size
    size_t k_out_size_; // num_k_head * k_dim / tp_size
    size_t v_out_size_; // num_v_head * v_dim / tp_size
};

class GateUpParallelLinear : public infinicore::nn::ColumnParallelLinear {
public:
    /**
     * @deprecated This function is deprecated and will be REMOVED in the next major release (v0.2.0).
     *
     * ⚠️ DEVELOPMENT POLICY:
     *   - NO new development or feature additions permitted on this interface
     *   - Only critical bug fixes (security/stability) allowed until removal
     *   - All new code MUST migrate to the polymorphic overload below
     *
     * Replacement: Use the polymorphic overload of this same function name with updated signature
     * Reason: Legacy signature lacks support for dynamic quantization modes.
     * Removal target: v0.2.0 (Q2 2026)
     */
    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool bias = false,
                         const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                         const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization,
                         bool bias = false,
                         const infinicore::DataType &dtype = infinicore::DataType::F32,
                         const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                         std::shared_ptr<infinicore::quantization::BaseQuantization> quantization,
                         const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward_split(infinicore::Tensor &input);

    infinicore::nn::Parameter get_gate_weight() const;

    infinicore::nn::Parameter get_gate_weight_scale() const;

    infinicore::nn::Parameter get_gate_weight_zeros() const;

    infinicore::nn::Parameter get_gate_bias() const;

    infinicore::nn::Parameter get_up_weight() const;

    infinicore::nn::Parameter get_up_weight_scale() const;

    infinicore::nn::Parameter get_up_weight_zeros() const;

    infinicore::nn::Parameter get_up_bias() const;

    bool has_gate_bias() const;

    bool has_up_bias() const;

private:
    bool gate_bias_;
    bool up_bias_;
};

#define INFINILM_QKV_LINEAR_INIT(name, q_name, k_name, v_name, ...)                     \
    name##_ = std::make_shared<layers::QKVParallelLinear>(__VA_ARGS__);                 \
    this->register_parameter(std::string(q_name) + ".weight", name##_->get_q_weight()); \
    this->register_parameter(std::string(k_name) + ".weight", name##_->get_k_weight()); \
    this->register_parameter(std::string(v_name) + ".weight", name##_->get_v_weight()); \
    if (name##_->has_q_bias())                                                          \
        this->register_parameter(std::string(q_name) + ".bias", name##_->get_q_bias()); \
    if (name##_->has_k_bias())                                                          \
        this->register_parameter(std::string(k_name) + ".bias", name##_->get_k_bias()); \
    if (name##_->has_v_bias())                                                          \
        this->register_parameter(std::string(v_name) + ".bias", name##_->get_v_bias());

#define INFINILM_GATE_UP_LINEAR_INIT(name, gate_name, up_name, ...)                           \
    name##_ = std::make_shared<layers::GateUpParallelLinear>(__VA_ARGS__);                    \
    this->register_parameter(std::string(gate_name) + ".weight", name##_->get_gate_weight()); \
    this->register_parameter(std::string(up_name) + ".weight", name##_->get_up_weight());     \
    if (name##_->has_gate_bias())                                                             \
        this->register_parameter(std::string(gate_name) + ".bias", name##_->get_gate_bias()); \
    if (name##_->has_up_bias())                                                               \
        this->register_parameter(std::string(up_name) + ".bias", name##_->get_up_bias());

// ========================= QKV Quantization ==================================
#define INFINILM_QKV_LINEAR_W8A8_INIT(name, q_name, k_name, v_name, ...)                            \
    name##_ = std::make_shared<layers::QKVParallelLinear>(__VA_ARGS__);                             \
    this->register_parameter(std::string(q_name) + ".weight", name##_->get_q_weight());             \
    this->register_parameter(std::string(q_name) + ".weight_scale", name##_->get_q_weight_scale()); \
    this->register_parameter(std::string(k_name) + ".weight", name##_->get_k_weight());             \
    this->register_parameter(std::string(k_name) + ".weight_scale", name##_->get_k_weight_scale()); \
    this->register_parameter(std::string(v_name) + ".weight", name##_->get_v_weight());             \
    this->register_parameter(std::string(v_name) + ".weight_scale", name##_->get_v_weight_scale()); \
    if (name##_->has_q_bias())                                                                      \
        this->register_parameter(std::string(q_name) + ".bias", name##_->get_q_bias());             \
    if (name##_->has_k_bias())                                                                      \
        this->register_parameter(std::string(k_name) + ".bias", name##_->get_k_bias());             \
    if (name##_->has_v_bias())                                                                      \
        this->register_parameter(std::string(v_name) + ".bias", name##_->get_v_bias());

#define INFINILM_QKV_LINEAR_W4A16AWQ_INIT(name, q_name, k_name, v_name, ...)                  \
    name##_ = std::make_shared<layers::QKVParallelLinear>(__VA_ARGS__);                       \
    this->register_parameter(std::string(q_name) + ".qweight", name##_->get_q_weight());      \
    this->register_parameter(std::string(q_name) + ".qzeros", name##_->get_q_weight_zeros()); \
    this->register_parameter(std::string(q_name) + ".scales", name##_->get_q_weight_scale()); \
    this->register_parameter(std::string(k_name) + ".qweight", name##_->get_k_weight());      \
    this->register_parameter(std::string(k_name) + ".qzeros", name##_->get_k_weight_zeros()); \
    this->register_parameter(std::string(k_name) + ".scales", name##_->get_k_weight_scale()); \
    this->register_parameter(std::string(v_name) + ".qweight", name##_->get_v_weight());      \
    this->register_parameter(std::string(v_name) + ".qzeros", name##_->get_v_weight_zeros()); \
    this->register_parameter(std::string(v_name) + ".scales", name##_->get_v_weight_scale()); \
    if (name##_->has_q_bias())                                                                \
        this->register_parameter(std::string(q_name) + ".bias", name##_->get_q_bias());       \
    if (name##_->has_k_bias())                                                                \
        this->register_parameter(std::string(k_name) + ".bias", name##_->get_k_bias());       \
    if (name##_->has_v_bias())                                                                \
        this->register_parameter(std::string(v_name) + ".bias", name##_->get_v_bias());

// ========================= Gate-Up Quantization ==============================
#define INFINILM_GATE_UP_LINEAR_W8A8_INIT(name, gate_name, up_name, ...)                                  \
    name##_ = std::make_shared<layers::GateUpParallelLinear>(__VA_ARGS__);                                \
    this->register_parameter(std::string(gate_name) + ".weight", name##_->get_gate_weight());             \
    this->register_parameter(std::string(gate_name) + ".weight_scale", name##_->get_gate_weight_scale()); \
    this->register_parameter(std::string(up_name) + ".weight", name##_->get_up_weight());                 \
    this->register_parameter(std::string(up_name) + ".weight_scale", name##_->get_up_weight_scale());     \
    if (name##_->has_gate_bias())                                                                         \
        this->register_parameter(std::string(gate_name) + ".bias", name##_->get_gate_bias());             \
    if (name##_->has_up_bias())                                                                           \
        this->register_parameter(std::string(up_name) + ".bias", name##_->get_up_bias());

#define INFINILM_GATE_UP_LINEAR_W4A16AWQ_INIT(name, gate_name, up_name, ...)                        \
    name##_ = std::make_shared<layers::GateUpParallelLinear>(__VA_ARGS__);                          \
    this->register_parameter(std::string(gate_name) + ".qweight", name##_->get_gate_weight());      \
    this->register_parameter(std::string(gate_name) + ".scales", name##_->get_gate_weight_scale()); \
    this->register_parameter(std::string(gate_name) + ".qzeros", name##_->get_gate_weight_zeros()); \
    this->register_parameter(std::string(up_name) + ".qweight", name##_->get_up_weight());          \
    this->register_parameter(std::string(up_name) + ".scales", name##_->get_up_weight_scale());     \
    this->register_parameter(std::string(up_name) + ".qzeros", name##_->get_up_weight_zeros());     \
    if (name##_->has_gate_bias())                                                                   \
        this->register_parameter(std::string(gate_name) + ".bias", name##_->get_gate_bias());       \
    if (name##_->has_up_bias())                                                                     \
        this->register_parameter(std::string(up_name) + ".bias", name##_->get_up_bias());
} // namespace infinilm::layers
