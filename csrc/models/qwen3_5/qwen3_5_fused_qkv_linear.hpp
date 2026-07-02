#pragma once

#include "../../layers/linear/linear.hpp"

namespace infinilm::models::qwen3_5 {

class Qwen35FusedQKVLinear : public infinilm::nn::ColumnParallelLinear {
public:
    Qwen35FusedQKVLinear(size_t hidden_size,
                         size_t head_dim,
                         size_t num_q_head,
                         size_t num_kv_head,
                         const std::string &q_name,
                         const std::string &k_name,
                         const std::string &v_name,
                         infinilm::layers::linear::RegisterParamFn register_fn,
                         std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = nullptr,
                         bool bias = false,
                         const infinicore::DataType &dtype = infinicore::DataType::F32,
                         const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    void process_weights_after_loading() override;

    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    forward_split(infinicore::Tensor &input);

private:
    static size_t calculate_kv_replicas(size_t num_kv_head, size_t tp_size) {
        if (num_kv_head % tp_size == 0) {
            return 1;
        }
        if (tp_size % num_kv_head == 0) {
            return (tp_size + num_kv_head - 1) / num_kv_head;
        }
        throw std::runtime_error("Invalid KV head configuration");
    }

    size_t head_dim_;
    size_t local_num_q_heads_;
    size_t q_proj_out_size_;
    size_t q_out_size_;
    size_t k_out_size_;
    size_t v_out_size_;
    size_t num_kv_head_;
    infinilm::layers::linear::RegisterParamFn register_fn_;
    std::vector<infinilm::quantization::SplitInfo> split_infos_;
};

} // namespace infinilm::models::qwen3_5
