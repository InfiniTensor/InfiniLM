#include "fused_linear.hpp"

#include "../../utils/agent_debug.hpp"
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
                        dtype, device, rank_info) {}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : infinilm::nn::ColumnParallelLinear(
          hidden_size,
          calculate_out_feature_size(num_q_head, q_dim, num_k_head, k_dim, num_v_head, v_dim, rank_info),
          quantization,
          (q_bias || k_bias || v_bias),
          dtype,
          device,
          rank_info.tp_rank,
          rank_info.tp_size),
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

infinicore::Tensor QKVParallelLinear::q_weight() const {
    return weight_for_prefix("q_proj");
}

infinicore::Tensor QKVParallelLinear::k_weight() const {
    return weight_for_prefix("k_proj");
}

infinicore::Tensor QKVParallelLinear::v_weight() const {
    return weight_for_prefix("v_proj");
}

infinicore::Tensor QKVParallelLinear::weight_for_prefix(const std::string &prefix) const {
    for (const auto &suffix : {"weight", "qweight"}) {
        auto it = parameters_.find(prefix + "." + suffix);
        if (it != parameters_.end()) {
            return static_cast<const infinicore::Tensor &>(it->second);
        }
    }
    if (!split_infos_.empty()) {
        for (const auto &s : split_infos_) {
            if (s.prefix != prefix) {
                continue;
            }
            const std::string fused_key = parameters_.count("qweight") ? "qweight" : "weight";
            const auto &fused = get_parameter_ref(fused_key);
            return fused->narrow({{static_cast<size_t>(split_dim_), s.start, s.size}});
        }
    }
    throw std::runtime_error("QKVParallelLinear: missing weight for " + prefix);
}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head, size_t num_kv_head,
                                     const std::string &q_name, const std::string &k_name, const std::string &v_name,
                                     RegisterParamFn register_fn,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size, head_dim, num_q_head, num_kv_head, quantization, bias, dtype, device, rank_info) {
    register_fn_ = register_fn;
    split_infos_ = {
        {q_name, 0, q_out_size_, 0},
        {k_name, q_out_size_, k_out_size_, num_k_head_},
        {v_name, q_out_size_ + k_out_size_, v_out_size_, num_v_head_},
    };
    auto params = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     const std::string &q_name, const std::string &k_name, const std::string &v_name,
                                     RegisterParamFn register_fn,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size, q_dim, k_dim, v_dim, num_q_head, num_k_head, num_v_head,
                        q_bias, k_bias, v_bias, quantization, dtype, device, rank_info) {
    register_fn_ = register_fn;
    split_infos_ = {
        {q_name, 0, q_out_size_, 0},
        {k_name, q_out_size_, k_out_size_, num_k_head_},
        {v_name, q_out_size_ + k_out_size_, v_out_size_, num_v_head_},
    };
    auto params = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

void QKVParallelLinear::process_weights_after_loading() {
    // #region agent log
    static thread_local bool qkv_pw_logged = false;
    // #endregion
    infinilm::quantization::ParamsMap params;
    for (const auto &[name, param] : parameters_) {
        params[name] = static_cast<const infinicore::Tensor &>(param);
    }

    auto new_quant = quantization_->process_weights_after_loading(params, device_);
    // #region agent log
    if (!qkv_pw_logged) {
        infinilm::agent_debug::log(
            "fused_linear.cpp:QKVParallelLinear::process_weights_after_loading",
            "qkv_process_weights",
            "F",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank_) +
                ",\"tp_size\":" + std::to_string(tp_size_) +
                ",\"repacked\":" + (new_quant ? "true" : "false") + "}",
            "post-fix");
        qkv_pw_logged = true;
    }
    // #endregion
    if (!new_quant) {
        // Fused q/k/v were registered at construction and loaded directly from
        // checkpoint; re-splitting from the unused fused `weight` buffer would
        // clobber TP-sharded q_proj/k_proj/v_proj (tp>4 paged eager regression).
        return;
    }

    for (auto &[name, param] : parameters_) {
        param = infinicore::nn::Parameter();
    }

    for (const auto &[name, tensor] : params) {
        auto it = parameters_.find(name);
        if (it == parameters_.end()) {
            continue;
        }
        it->second = infinicore::nn::Parameter(tensor);
    }

    quantization_ = std::move(new_quant);

    if (register_fn_ && !split_infos_.empty()) {
        auto split = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
        for (auto &sp : split) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }
}

// ---------------------------------------------------------
// Gate-Up Parallel Linear
// ---------------------------------------------------------
GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : GateUpParallelLinear(hidden_size, intermediate_size, bias, bias, quantization, dtype, device, rank_info) {
}

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : infinilm::nn::ColumnParallelLinear(hidden_size, intermediate_size * 2, quantization, gate_bias || up_bias, dtype, device, rank_info.tp_rank, rank_info.tp_size), gate_bias_(gate_bias), up_bias_(up_bias) {
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
                                           engine::distributed::RankInfo rank_info)
    : GateUpParallelLinear(hidden_size, intermediate_size, quantization, bias, dtype, device, rank_info) {
    const std::string &key_name = parameters_.count("qweight") ? "qweight" : "weight";
    const auto &key_param = get_parameter_ref(key_name);
    int fused_dim = this->get_quantization()->get_fused_split_dim();
    size_t logical_output = this->get_quantization()->get_logical_dim_size(key_param->size(fused_dim));
    size_t half_size = logical_output / 2;
    register_fn_ = register_fn;
    split_infos_ = {
        {gate_name, 0, half_size},
        {up_name, half_size, half_size},
    };
    auto params = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

void GateUpParallelLinear::process_weights_after_loading() {
    infinilm::quantization::ParamsMap params;
    for (const auto &[name, param] : parameters_) {
        params[name] = static_cast<const infinicore::Tensor &>(param);
    }

    auto new_quant = quantization_->process_weights_after_loading(params, device_);
    if (!new_quant) {
        return;
    }

    for (auto &[name, param] : parameters_) {
        param = infinicore::nn::Parameter();
    }

    for (const auto &[name, tensor] : params) {
        auto it = parameters_.find(name);
        if (it == parameters_.end()) {
            continue;
        }
        it->second = infinicore::nn::Parameter(tensor);
    }

    quantization_ = std::move(new_quant);

    if (register_fn_ && !split_infos_.empty()) {
        auto split = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
        for (auto &sp : split) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }
}

} // namespace infinilm::layers::linear
