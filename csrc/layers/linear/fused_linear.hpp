#pragma once
#include "../../engine/distributed/communication_group.hpp"
#include "../quantization/quantization.hpp"
#include "linear.hpp"
#include <functional>

namespace infinilm::layers::linear {
using RegisterParamFn = std::function<void(const std::string &, infinicore::nn::Parameter)>;

class QKVParallelLinear : public infinilm::nn::ColumnParallelLinear {
public:
    explicit QKVParallelLinear(size_t hidden_size,
                               size_t q_dim, size_t k_dim, size_t v_dim,
                               size_t num_q_head, size_t num_k_head, size_t num_v_head,
                               bool q_bias, bool k_bias, bool v_bias,
                               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = nullptr,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    explicit QKVParallelLinear(size_t hidden_size,
                               size_t head_dim,
                               size_t num_q_head, size_t num_kv_head,
                               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = nullptr,
                               bool bias = false,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    QKVParallelLinear(size_t hidden_size,
                      size_t head_dim,
                      size_t num_q_head, size_t num_kv_head,
                      const std::string &q_name, const std::string &k_name, const std::string &v_name,
                      RegisterParamFn register_fn,
                      std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = nullptr,
                      bool bias = false,
                      const infinicore::DataType &dtype = infinicore::DataType::F32,
                      const infinicore::Device &device = infinicore::Device(),
                      engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    void process_weights_after_loading() override;

    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    forward_split(infinicore::Tensor &input);

    bool has_q_bias() const;
    bool has_k_bias() const;
    bool has_v_bias() const;

private:
    static size_t calculate_kv_replicas(size_t num_k_head, size_t tp_size) {
        if (num_k_head % tp_size == 0) {
            return 1;
        }
        if (tp_size % num_k_head == 0) {
            return (tp_size + num_k_head - 1) / num_k_head;
        }
        throw std::runtime_error("Invalid KV head configuration");
    }

    static size_t
    calculate_out_feature_size(size_t num_q_head, size_t q_dim, size_t num_k_head, size_t k_dim, size_t num_v_head, size_t v_dim, engine::distributed::RankInfo rank_info) {
        return num_q_head * q_dim + num_k_head * k_dim * calculate_kv_replicas(num_k_head, rank_info.tp_size) + num_v_head * v_dim * calculate_kv_replicas(num_v_head, rank_info.tp_size);
    }

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
    size_t q_out_size_;
    size_t k_out_size_;
    size_t v_out_size_;

    size_t num_kv_head_replicas_ = 1;
    RegisterParamFn register_fn_;
    std::vector<infinilm::quantization::SplitInfo> split_infos_;
};

class GateUpParallelLinear : public infinilm::nn::ColumnParallelLinear {
public:
    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size,
                         std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = nullptr,
                         bool bias = false,
                         const infinicore::DataType &dtype = infinicore::DataType::F32,
                         const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                         std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = nullptr,
                         const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size,
                         const std::string &gate_name, const std::string &up_name,
                         RegisterParamFn register_fn,
                         std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = nullptr,
                         bool bias = false,
                         const infinicore::DataType &dtype = infinicore::DataType::F32,
                         const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    void process_weights_after_loading() override;

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward_split(infinicore::Tensor &input);

    bool has_gate_bias() const;
    bool has_up_bias() const;

private:
    bool gate_bias_;
    bool up_bias_;
    RegisterParamFn register_fn_;
    std::vector<infinilm::quantization::SplitInfo> split_infos_;
};

} // namespace infinilm::layers::linear
