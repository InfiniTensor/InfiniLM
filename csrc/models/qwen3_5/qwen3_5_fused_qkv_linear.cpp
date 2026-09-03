#include "qwen3_5_fused_qkv_linear.hpp"

namespace infinilm::models::qwen3_5 {

Qwen35FusedQKVLinear::Qwen35FusedQKVLinear(size_t hidden_size,
                                           size_t head_dim,
                                           size_t num_q_head,
                                           size_t num_kv_head,
                                           const std::string &q_name,
                                           const std::string &k_name,
                                           const std::string &v_name,
                                           infinilm::layers::linear::RegisterParamFn register_fn,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           bool bias,
                                           const infinicore::DataType &dtype,
                                           const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info,
                                           const std::string &prefix)
    : infinilm::layers::linear::ColumnParallelLinear(
        hidden_size,
        num_q_head * head_dim * 2 + num_kv_head * head_dim * calculate_kv_replicas(num_kv_head, rank_info.tp_size) * 2,
        quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
        bias,
        dtype,
        device,
        rank_info.tp_rank,
        rank_info.tp_size,
        -1,
        prefix),
      head_dim_(head_dim),
      local_num_q_heads_(num_q_head / tp_size_),
      q_proj_out_size_(num_q_head * head_dim * 2 / tp_size_),
      q_out_size_(num_q_head * head_dim / tp_size_),
      k_out_size_(calculate_kv_replicas(num_kv_head, rank_info.tp_size) * num_kv_head * head_dim / tp_size_),
      v_out_size_(calculate_kv_replicas(num_kv_head, rank_info.tp_size) * num_kv_head * head_dim / tp_size_),
      num_kv_head_(num_kv_head),
      register_fn_(register_fn) {
    if (this->sharded_) {
        // GGUF：三段各有自己的 checkpoint 张量（q_proj 含交错的 gate），不存在可 narrow
        // 的融合 buffer；stem 必须带结尾的 '.'，与类型表里的张量名逐字相等。
        if (prefix.empty()) {
            throw std::runtime_error(
                "Qwen35FusedQKVLinear: GGUF 量化必须传 layer prefix（形如 layers.3.self_attn）");
        }
        shard_specs_ = {
            {q_name, q_proj_out_size_, prefix + "." + q_name + "."},
            {k_name, k_out_size_, prefix + "." + k_name + "."},
            {v_name, v_out_size_, prefix + "." + v_name + "."},
        };
    } else {
        split_infos_ = {
            {q_name, 0, q_proj_out_size_, 0},
            {k_name, q_proj_out_size_, k_out_size_, num_kv_head_},
            {v_name, q_proj_out_size_ + k_out_size_, v_out_size_, num_kv_head_},
        };
    }
    register_fused_params();
}

void Qwen35FusedQKVLinear::register_fused_params() {
    if (!register_fn_) {
        return;
    }
    // GGUF 分支：逐 shard 一次 GEMM，forward() 里拼回同一根 [B,S,q|k|v]，
    // 所以下面 forward_split() 的 narrow 偏移量不用改。
    auto params = this->sharded_
                    ? this->init_fused_shards(shard_specs_)
                    : this->split_params(split_infos_, tp_rank_, tp_size_, num_kv_head_);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
Qwen35FusedQKVLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);
    auto shape = output->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto q_proj_out = output->narrow({{2, 0, q_proj_out_size_}});
    auto q_proj_heads = q_proj_out->view({batch_size, seq_len, local_num_q_heads_, head_dim_ * 2});
    auto q_out = q_proj_heads->narrow({{3, 0, head_dim_}});
    auto gate_out = q_proj_heads->narrow({{3, head_dim_, head_dim_}});
    auto k_out = output->narrow({{2, q_proj_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_proj_out_size_ + k_out_size_, v_out_size_}});

    return std::make_tuple(q_out, gate_out, k_out, v_out);
}

void Qwen35FusedQKVLinear::process_weights_after_loading() {
    BaseLinear::process_weights_after_loading();
    // sharded_（GGUF）时 split_infos_ 为空：那些 shard 参数就是加载目标，
    // 重新分配会把已读进来的块字节丢掉
    if (register_fn_ && !split_infos_.empty()) {
        register_fused_params();
    }
}

} // namespace infinilm::models::qwen3_5
