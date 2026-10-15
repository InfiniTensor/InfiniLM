#include "fused_linear.hpp"

#include <spdlog/spdlog.h>

namespace infinilm::layers::linear {
namespace {

size_t merged_output_size(const std::vector<size_t> &output_sizes) {
    size_t total = 0;
    for (const auto size : output_sizes) {
        total += size;
    }
    return total;
}

} // namespace

// ---------------------------------------------------------
// Merged Replicated Linear
// ---------------------------------------------------------
MergedReplicatedLinear::MergedReplicatedLinear(
    size_t input_size,
    const std::vector<size_t> &output_sizes,
    const std::vector<std::string> &param_names,
    RegisterParamFn register_fn,
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
    bool bias,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : infinilm::nn::Linear(
        input_size,
        merged_output_size(output_sizes),
        quantization == nullptr
            ? std::make_shared<infinilm::quantization::NoneQuantization>()
            : quantization,
        bias,
        dtype,
        device),
      output_sizes_(output_sizes),
      register_fn_(std::move(register_fn)) {
    if (output_sizes_.empty() || output_sizes_.size() != param_names.size()) {
        throw std::runtime_error(
            "MergedReplicatedLinear expects non-empty, equally sized output_sizes and param_names");
    }
    size_t offset = 0;
    for (size_t i = 0; i < output_sizes_.size(); ++i) {
        split_infos_.push_back({param_names[i], offset, output_sizes_[i]});
        offset += output_sizes_[i];
    }
    auto params = this->split_params(split_infos_, 0, 1, -1);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

std::vector<infinicore::Tensor>
MergedReplicatedLinear::forward_split(infinicore::Tensor &input) const {
    auto output = this->forward(input);
    const size_t dim = output->ndim() - 1;
    std::vector<infinicore::Tensor> result;
    result.reserve(output_sizes_.size());
    size_t offset = 0;
    for (const auto size : output_sizes_) {
        result.push_back(output->narrow({{dim, offset, size}}));
        offset += size;
    }
    return result;
}

void MergedReplicatedLinear::process_weights_after_loading() {
    BaseLinear::process_weights_after_loading();
    auto params = this->split_params(split_infos_, 0, 1, -1);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }
}

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
        quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
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
    : QKVParallelLinear(hidden_size, head_dim, head_dim, head_dim, num_q_head, num_kv_head, num_kv_head, bias, bias, bias, q_name, k_name, v_name, register_fn, quantization, dtype, device, rank_info) {
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
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size, q_dim, k_dim, v_dim, num_q_head, num_k_head, num_v_head, q_bias, k_bias, v_bias, quantization, dtype, device, rank_info) {
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
    BaseLinear::process_weights_after_loading();
    if (register_fn_ && !split_infos_.empty()) {
        auto params = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
        for (auto &sp : params) {
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
    : infinilm::nn::ColumnParallelLinear(
        hidden_size,
        intermediate_size * 2,
        quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
        gate_bias || up_bias,
        dtype,
        device,
        rank_info.tp_rank,
        rank_info.tp_size),
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
    BaseLinear::process_weights_after_loading();
    if (register_fn_ && !split_infos_.empty()) {
        auto params = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
        for (auto &sp : params) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }
}

} // namespace infinilm::layers::linear
