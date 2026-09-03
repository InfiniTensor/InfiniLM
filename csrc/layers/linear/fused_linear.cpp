#include "fused_linear.hpp"

#include <spdlog/spdlog.h>

namespace infinilm::layers::linear {
// ---------------------------------------------------------
// QKV Parallel Linear
// ---------------------------------------------------------
QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head,
                                     size_t num_kv_head,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size,
                        head_dim, head_dim, head_dim,
                        num_q_head, num_kv_head, num_kv_head,
                        bias, bias, bias,
                        quantization,
                        dtype, device, rank_info, "") {}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info,
                                     const std::string &stem)
    : infinilm::nn::ColumnParallelLinear(
          hidden_size,
          calculate_out_feature_size(num_q_head, q_dim, num_k_head, k_dim, num_v_head, v_dim, rank_info),
          quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
          (q_bias || k_bias || v_bias),
          dtype,
          device,
          rank_info.tp_rank,
          rank_info.tp_size,
          -1,
          stem),
      q_dim_(q_dim),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_q_head_(num_q_head),
      num_k_head_(num_k_head),
      num_v_head_(num_v_head),
      q_bias_(q_bias),
      k_bias_(k_bias),
      v_bias_(v_bias),
      num_kv_head_replicas_(calculate_kv_replicas(num_k_head, rank_info.tp_size)) {

    if ((q_bias_ != k_bias_) || (k_bias_ != v_bias_)) {
        throw std::runtime_error("q_bias, k_bias, v_bias must all match");
    }

    q_out_size_ = num_q_head_ * q_dim_ / tp_size_;
    k_out_size_ = num_kv_head_replicas_ * num_k_head_ * k_dim_ / tp_size_;
    v_out_size_ = num_kv_head_replicas_ * num_v_head_ * v_dim_ / tp_size_;
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
QKVParallelLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);

    auto q_out = output->narrow({{2, 0, q_out_size_}});
    auto k_out = output->narrow({{2, q_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_out_size_ + k_out_size_, v_out_size_}});

    return std::make_tuple(q_out, k_out, v_out);
}

bool QKVParallelLinear::has_q_bias() const { return q_bias_; }
bool QKVParallelLinear::has_k_bias() const { return k_bias_; }
bool QKVParallelLinear::has_v_bias() const { return v_bias_; }

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head, size_t num_kv_head,
                                     const std::string &q_name, const std::string &k_name, const std::string &v_name,
                                     RegisterParamFn register_fn,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info,
                                     const std::string &prefix)
    : QKVParallelLinear(hidden_size, head_dim, head_dim, head_dim, num_q_head, num_kv_head, num_kv_head, bias, bias, bias, q_name, k_name, v_name, register_fn, quantization, dtype, device, rank_info, prefix) {
}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     const std::string &q_name, const std::string &k_name, const std::string &v_name,
                                     RegisterParamFn register_fn,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info,
                                     const std::string &prefix)
    : QKVParallelLinear(hidden_size, q_dim, k_dim, v_dim, num_q_head, num_k_head, num_v_head, q_bias, k_bias, v_bias, quantization, dtype, device, rank_info, prefix) {
    register_fn_ = register_fn;
    if (this->sharded_) {
        // GGUF：q/k/v 在本文件里 ggml 类型全不相同（§6.0 纠正 1），没有可 narrow 的
        // 融合 buffer —— 每 shard 各自一块，stem 指向各自的 checkpoint 张量。
        if (prefix.empty()) {
            throw std::runtime_error(
                "QKVParallelLinear: 按 checkpoint 张量名查表的量化方案（GGUF）必须传 prefix");
        }
        shard_specs_ = {
            {q_name, q_out_size_, prefix + "." + q_name + "."},
            {k_name, k_out_size_, prefix + "." + k_name + "."},
            {v_name, v_out_size_, prefix + "." + v_name + "."},
        };
    } else {
        split_infos_ = {
            {q_name, 0, q_out_size_, 0},
            {k_name, q_out_size_, k_out_size_, num_k_head_},
            {v_name, q_out_size_ + k_out_size_, v_out_size_, num_v_head_},
        };
    }
    register_fused_params();
}

void QKVParallelLinear::register_fused_params() {
    if (!register_fn_) {
        return;
    }
    auto params = this->sharded_
                    ? this->init_fused_shards(shard_specs_)
                    : this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

void QKVParallelLinear::process_weights_after_loading() {
    BaseLinear::process_weights_after_loading();
    // 融合量化布局（sharded_）下 split_infos_ 为空，不会重跑：那些 shard 参数就是
    // 加载目标，重新分配会把已读进来的字节丢掉
    if (register_fn_ && !split_infos_.empty()) {
        register_fused_params();
    }
}

// ---------------------------------------------------------
// Gate-Up Parallel Linear
// ---------------------------------------------------------
GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info,
                                           const std::string &stem)
    : GateUpParallelLinear(hidden_size, intermediate_size, bias, bias, quantization, dtype, device, rank_info, stem) {
}

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info,
                                           const std::string &stem)
    : infinilm::nn::ColumnParallelLinear(
          hidden_size,
          intermediate_size * 2,
          quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
          gate_bias || up_bias,
          dtype,
          device,
          rank_info.tp_rank,
          rank_info.tp_size,
          -1,
          stem),
      gate_bias_(gate_bias),
      up_bias_(up_bias) {
    if (gate_bias_ != up_bias_) {
        throw std::runtime_error("Not supported yet: gate_bias and up_bias should be given at the same time");
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> GateUpParallelLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);
    auto cols = output->shape()[2];
    auto gate_output = output->narrow({{2, 0, cols / 2}});
    auto up_output = output->narrow({{2, cols / 2, cols / 2}});
    return std::make_tuple(gate_output, up_output);
}

bool GateUpParallelLinear::has_gate_bias() const { return gate_bias_; }
bool GateUpParallelLinear::has_up_bias() const { return up_bias_; }

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size,
                                           const std::string &gate_name, const std::string &up_name,
                                           RegisterParamFn register_fn,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info,
                                           const std::string &prefix)
    : GateUpParallelLinear(hidden_size, intermediate_size, quantization, bias, dtype, device, rank_info, prefix) {
    register_fn_ = register_fn;
    if (this->sharded_) {
        // GGUF：gate/up 在本文件 28/64 层类型不同（§6.0 纠正 1），两者 row_bytes 不同，
        // 装不进同一块融合 buffer，所以各自一块、各自查自己是 blob 还是稠密。
        if (prefix.empty()) {
            throw std::runtime_error(
                "GateUpParallelLinear: 按 checkpoint 张量名查表的量化方案（GGUF）必须传 prefix");
        }
        const size_t half = intermediate_size / tp_size_;
        shard_specs_ = {
            {gate_name, half, prefix + "." + gate_name + "."},
            {up_name, half, prefix + "." + up_name + "."},
        };
    } else {
        const std::string &key_name = parameters_.count("qweight") ? "qweight" : "weight";
        const auto &key_param = get_parameter_ref(key_name);
        int fused_dim = this->get_quantization()->get_fused_split_dim();
        size_t logical_output = this->get_quantization()->get_logical_dim_size(key_param->size(fused_dim));
        size_t half_size = logical_output / 2;
        split_infos_ = {
            {gate_name, 0, half_size},
            {up_name, half_size, half_size},
        };
    }
    register_fused_params();
}

void GateUpParallelLinear::register_fused_params() {
    if (!register_fn_) {
        return;
    }
    auto params = this->sharded_
                    ? this->init_fused_shards(shard_specs_)
                    : this->split_params(split_infos_, tp_rank_, tp_size_, -1);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

void GateUpParallelLinear::process_weights_after_loading() {
    BaseLinear::process_weights_after_loading();
    // 同 QKVParallelLinear：sharded_ 时 split_infos_ 为空，不重跑切分
    if (register_fn_ && !split_infos_.empty()) {
        register_fused_params();
    }
}

} // namespace infinilm::layers::linear
