#include "fused_linear.hpp"

#include "../../global_state/global_state.hpp"
#include <cstdlib>
#include <cstring>
#include <infinicore/ops/ascend_format_cast.hpp>
#include <infinicore/ops/cat.hpp>
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
    auto output = forward_shared_kv(input);

    auto q_out = output->narrow({{2, 0, q_out_size_}});
    auto k_out = output->narrow({{2, q_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_out_size_ + k_out_size_, v_out_size_}});

    return std::make_tuple(q_out, k_out, v_out);
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
QKVParallelLinear::forward_split_with_qk(infinicore::Tensor &input) {
    auto output = forward_shared_kv(input);

    auto q_out = output->narrow({{2, 0, q_out_size_}});
    auto k_out = output->narrow({{2, q_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_out_size_ + k_out_size_, v_out_size_}});
    // q and k are adjacent contiguous slices of `output`; expose a merged
    // view so attention can apply RoPE to q+k in a single kernel launch.
    auto qk_out = output->narrow({{2, 0, q_out_size_ + k_out_size_}});

    return std::make_tuple(q_out, k_out, v_out, qk_out);
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

void QKVParallelLinear::prepare_shared_kv_projection() {
    using namespace infinicore;
    const char *flag = std::getenv("INFINILM_ASCEND_SHARE_KV_PROJECTION");
    if (!flag || std::strcmp(flag, "1") != 0 || shared_kv_weight_
        || device_.getType() != Device::Type::ASCEND || dtype_ != DataType::BF16
        || !global_state::get_infinilm_config().pre_transpose
        || !std::dynamic_pointer_cast<infinilm::quantization::NoneQuantization>(quantization_)
        || q_dim_ != 128 || k_dim_ != q_dim_ || v_dim_ != q_dim_
        || num_k_head_ != num_q_head_ || num_v_head_ != num_q_head_) {
        return;
    }
    auto original = weight();
    const size_t columns = q_out_size_ + k_out_size_ + v_out_size_;
    if (original->shape() != Shape{columns, in_features_}
        || !original->is_contiguous() || k_out_size_ != v_out_size_
        || k_out_size_ % k_dim_ || k_out_size_ / k_dim_ < 2) {
        return;
    }
    context::syncStream();
    auto host_weight = original->to(Device::cpu());
    std::optional<Tensor> host_bias;
    if (has_bias_) {
        auto b = bias();
        if (b->shape() != Shape{columns} || !b->is_contiguous() || b->dtype() != dtype_) {
            return;
        }
        host_bias = b->to(Device::cpu());
    }
    context::syncStream();
    const auto *w = static_cast<const std::byte *>(host_weight->data());
    const auto *b = host_bias ? static_cast<const std::byte *>(host_bias.value()->data()) : nullptr;
    constexpr size_t bytes_per_value = 2;
    const size_t heads = k_out_size_ / k_dim_;
    size_t repeats = 1;
    // Compare every byte of every head and bias, not a sampled hash. Unequal
    // heads keep the original projection. This does not alter attention heads.
    for (size_t candidate = heads; candidate >= 2; --candidate) {
        if (heads % candidate) {
            continue;
        }
        bool same = true;
        for (size_t base : {q_out_size_, q_out_size_ + k_out_size_}) {
            for (size_t group = 0; group < heads && same; group += candidate) {
                const size_t first = base + group * k_dim_;
                for (size_t i = 1; i < candidate; ++i) {
                    const size_t row = first + i * k_dim_;
                    if (std::memcmp(w + first * in_features_ * bytes_per_value,
                                    w + row * in_features_ * bytes_per_value,
                                    k_dim_ * in_features_ * bytes_per_value)
                            != 0
                        || (b && std::memcmp(b + first * bytes_per_value, b + row * bytes_per_value, k_dim_ * bytes_per_value) != 0)) {
                        same = false;
                        break;
                    }
                }
            }
        }
        if (same) {
            repeats = candidate;
            break;
        }
    }
    if (repeats == 1) {
        return;
    }
    context::setDevice(device_);
    std::vector<Tensor> rows{original->narrow({{0, 0, q_out_size_}})};
    for (size_t base : {q_out_size_, q_out_size_ + k_out_size_}) {
        for (size_t head = 0; head < heads; head += repeats) {
            rows.push_back(original->narrow({{0, base + head * k_dim_, k_dim_}}));
        }
    }
    auto selected = op::cat(rows, 0);
    auto packed = selected->permute({1, 0})->contiguous();
    shared_kv_weight_ = op::ascend_format_cast_nz(packed);
    shared_kv_repeat_ = repeats;
    context::syncStream();
    spdlog::info("Ascend shared KV projection verified: rank={}, local_kv_heads={}, repeats={}, projected_features={}, restored_features={}",
                 tp_rank_, heads, repeats, q_out_size_ + 2 * k_out_size_ / repeats, columns);
}

infinicore::Tensor QKVParallelLinear::forward_shared_kv(infinicore::Tensor &input) {
    using namespace infinicore;
    if (!shared_kv_weight_ || input->ndim() != 3 || input->size(0) != 1
        || input->size(1) <= 16 || input->size(1) > 2048
        || input->dtype() != dtype_ || input->device() != device_) {
        return this->forward(input);
    }
    const size_t tokens = input->size(1);
    const size_t compact_k = k_out_size_ / shared_kv_repeat_;
    const size_t compact_columns = q_out_size_ + 2 * compact_k;
    const size_t columns = q_out_size_ + k_out_size_ + v_out_size_;
    auto projected = op::linear_packed_nz(input->contiguous(), shared_kv_weight_.value(), std::nullopt, alpha_);
    auto output = Tensor::empty({1, tokens, columns}, dtype_, device_);
    output->narrow({{2, 0, q_out_size_}})->copy_from(projected->narrow({{2, 0, q_out_size_}}));
    // Expand only proven-identical linear results. All 64 logical MHA heads
    // are present before bias, RoPE, cache writes and attention execute.
    for (size_t part = 0; part < 2; ++part) {
        auto source = projected->narrow({{2, q_out_size_ + part * compact_k, compact_k}})
                          ->as_strided({tokens, compact_k / k_dim_, shared_kv_repeat_, k_dim_},
                                       {static_cast<ptrdiff_t>(compact_columns), static_cast<ptrdiff_t>(k_dim_), 0, 1});
        auto destination = output->narrow({{2, q_out_size_ + part * k_out_size_, k_out_size_}})
                               ->as_strided(source->shape(),
                                            {static_cast<ptrdiff_t>(columns), static_cast<ptrdiff_t>(shared_kv_repeat_ * k_dim_), static_cast<ptrdiff_t>(k_dim_), 1});
        destination->copy_from(source);
    }
    if (has_bias_) {
        auto matrix = output->view({tokens, columns});
        op::add_(matrix, matrix, bias()->as_strided({tokens, columns}, {0, 1}));
    }
    return output;
}

void QKVParallelLinear::process_weights_after_loading() {
    prepare_shared_kv_projection();
    // Split first, while the weight is still in the checkpoint [OC, IC]
    // layout: the split_infos_ offsets are only valid there, and narrow
    // must not run on the converted [IC, OC] / FRACTAL_NZ weight.
    if (register_fn_ && !split_infos_.empty()) {
        auto params = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
        for (auto &sp : params) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }

    // Then convert: pre-transpose and optional FRACTAL_NZ cast.
    BaseLinear::process_weights_after_loading();

    // Re-split against the converted weight so the sub-parameter views stop
    // pinning the old [OC, IC] storage (otherwise device memory stays
    // doubled and tp=4 OOMs on FM9G-70B). The weight is [IC, OC] now;
    // split_params flips the narrow dim based on the recorded weight
    // layout. The refreshed views are not consumed by the inference path.
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
    // Split first, while the weight is still in the checkpoint [OC, IC]
    // layout: the split_infos_ offsets are only valid there, and narrow
    // must not run on the converted [IC, OC] / FRACTAL_NZ weight.
    if (register_fn_ && !split_infos_.empty()) {
        auto params = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
        for (auto &sp : params) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }

    // Then convert: pre-transpose and optional FRACTAL_NZ cast.
    BaseLinear::process_weights_after_loading();

    // Re-split against the converted weight so the sub-parameter views stop
    // pinning the old [OC, IC] storage (otherwise device memory stays
    // doubled and tp=4 OOMs on FM9G-70B). The weight is [IC, OC] now;
    // split_params flips the narrow dim based on the recorded weight
    // layout. The refreshed views are not consumed by the inference path.
    if (register_fn_ && !split_infos_.empty()) {
        auto params = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
        for (auto &sp : params) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }
}

} // namespace infinilm::layers::linear
