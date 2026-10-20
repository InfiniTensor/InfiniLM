#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "dense.hpp"
#include "base_linear.hpp"
#include "../quantization/w4_marlin_adapter.hpp"
#include "runtime/runtime.hpp"
#include <spdlog/spdlog.h>
#include <exception>
#include <climits>

namespace infinilm::nn {

namespace {

infinilm::runtime::DataType dense_output_dtype(
    const std::shared_ptr<infinilm::quantization::BaseQuantization> &quantization,
    const infinilm::runtime::Device &device, infinilm::runtime::DataType dtype, bool bias) {
    if (!quantization) throw std::invalid_argument("Linear quantization must be nonnull");
    if (quantization->get_quant_scheme() != infinilm::quantization::QuantScheme::NONE) return dtype;
    // Checked once at construction, never in a forward or parameter binding.
    const auto dense = std::dynamic_pointer_cast<const infinilm::quantization::NoneQuantization>(quantization);
    if (!dense) throw std::invalid_argument("NONE execution must be a dense binding");
    return infinilm::layers::linear::resolve_output_dtype(
        device, dtype, dtype, bias, dense->output_dtype());
}

} // namespace

BaseLinear::BaseLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features,
                       std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                       bool bias, const infinilm::runtime::DataType &dtype,
                       int split_dim, int tp_rank, int tp_size, int tp_num_heads)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias), dtype_(dtype),
      output_dtype_(dense_output_dtype(quantization, runtime.device(), dtype, bias)),
      split_dim_(split_dim), loading_tp_rank_(tp_rank), loading_tp_size_(tp_size),
      quantization_(std::move(quantization)) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Linear construction requires eager Runtime");
    if (!in_features || !out_features || !quantization_ || split_dim < -1 || split_dim > 1
        || tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size || tp_num_heads < -1)
        throw std::invalid_argument("Invalid explicit Linear construction geometry");
    device_ = runtime.device();
    const auto layout = quantization_->get_param_layout(
        in_features, out_features, split_dim, tp_rank, tp_size, tp_num_heads, dtype, bias);
    // Validate the entire layout before its first Parameter allocation.
    for (const auto &desc : layout) {
        if (desc.tp_size <= 0 || desc.tp_rank < 0 || desc.tp_rank >= desc.tp_size || desc.tp_num_heads < -1
            || desc.split_dim < -1 || (desc.split_dim >= 0 && static_cast<size_t>(desc.split_dim) >= desc.shape.size()))
            throw std::invalid_argument("Invalid explicit Linear parameter TP geometry");
        const auto shards = desc.tp_num_heads > 0 ? desc.tp_num_heads : desc.tp_size;
        if (desc.tp_size > 1 && desc.split_dim < desc.shape.size()
            && desc.shape[desc.split_dim] % shards != 0)
            throw std::invalid_argument("Explicit Linear parameter partition is not divisible");
    }
    infinilm::weights::Weight pending;
    try {
        for (const auto &desc : layout) {
            pending = infinilm::weights::Weight(runtime, desc.shape, desc.dtype,
            desc.split_dim, desc.tp_rank, desc.tp_size,
            desc.tp_num_heads >= 0 ? desc.tp_num_heads : 0);
            this->bind_weight(desc.name, pending);
        }
        bound_params_.reserve(local_weights_.size());
        for (const auto &[name, param] : local_weights_)
            bound_params_.emplace(name, static_cast<const infinilm::runtime::Tensor &>(param));
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor BaseLinear::compute_linear(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) const {
    runtime.assertOwnerThread();
    if (infinilm::quantization::is_w4(quantization_->get_quant_scheme())) {
        if (!w4_execution_) throw std::logic_error("W4 layer has no published explicit Runtime binding");
        return w4_execution_->forward(runtime, input, has_bias_, alpha_);
    }
    return quantization_->forward(runtime, bound_params_, input, has_bias_, alpha_);
}

infinilm::runtime::Tensor BaseLinear::compute_linear_allreduce(
    infinilm::runtime::Runtime &runtime,
    infinilm::runtime::Tensor &input,
    infinicclComm_t communicator) const {
    runtime.assertOwnerThread();
    if (output_dtype_ != dtype_)
        throw std::invalid_argument("mixed dense output is not qualified for allreduce");
    if (infinilm::quantization::is_w4(quantization_->get_quant_scheme())) {
        if (!w4_execution_) throw std::logic_error("W4 layer has no published explicit Runtime binding");
        return w4_execution_->forward_allreduce(runtime, input, has_bias_, communicator, alpha_);
    }
    return quantization_->forward_allreduce(
        runtime, bound_params_, input, has_bias_, communicator, alpha_);
}

infinilm::runtime::Tensor BaseLinear::forward(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) const {
    return compute_linear(runtime, input);
}

infinilm::runtime::Tensor BaseLinear::forward(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input,
                                     infinilm::runtime::Tensor &residual) const {
    runtime.assertOwnerThread();
    if (output_dtype_ != dtype_)
        throw std::invalid_argument("mixed dense output is not qualified for residual addition");
    if (infinilm::quantization::is_w4(quantization_->get_quant_scheme())) {
        runtime.assertOwnerThread();
        infinilm::runtime::Tensor output;
        try {
            output = compute_linear(runtime, input);
            infinilm::execution::submit<infini::ops::Add>(
                runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), output, residual, output);
            INFINILM_TRACE_OP(Add, output);
            return output;
        } catch (...) { infinilm::quantization::w4_detail::drain_or_terminate(runtime); throw; }
    }
    auto output = compute_linear(runtime, input);
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), output, residual, output);
    INFINILM_TRACE_OP(Add, output);
    return output;
}

void BaseLinear::validate_awq_metadata(infinilm::runtime::Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (quantization_->get_quant_scheme() != infinilm::quantization::QuantScheme::AWQ_W4A16)
        throw std::invalid_argument("AWQ preflight requires an AWQ layer");
    if (has_bias_ && (!bound_params_.count("bias") || !bound_params_.at("bias")))
        throw std::invalid_argument("AWQ layer requires its bias");
    const auto &config = quantization_->get_config();
    if (!config.is_object() || config.at("quant_method") != "awq")
        throw std::invalid_argument("AWQ preflight config differs from layer scheme");
    (void)infinilm::quantization::w4_detail::validate_metadata(runtime, bound_params_, config, split_dim_, false);
    }

std::optional<int64_t> BaseLinear::validate_gptq_metadata(
    infinilm::runtime::Runtime &runtime, bool retained_global_scale_table) const {
    runtime.assertOwnerThread();
    if (quantization_->get_quant_scheme() != infinilm::quantization::QuantScheme::GPTQ_W4A16)
        throw std::invalid_argument("GPTQ preflight requires a GPTQ layer");
    if (has_bias_ && (!bound_params_.count("bias") || !bound_params_.at("bias")))
        throw std::invalid_argument("GPTQ layer requires its bias");
    const auto &config = quantization_->get_config();
    if (!config.is_object() || config.at("quant_method") != "gptq")
        throw std::invalid_argument("GPTQ preflight config differs from layer scheme");
    const auto metadata = infinilm::quantization::w4_detail::validate_metadata(
        runtime, bound_params_, config, split_dim_, retained_global_scale_table);
    if (metadata.desc) return std::nullopt; // retain existing desc/partial layer-only domain
    size_t local_k = in_features_, expected = 0;
    if (split_dim_ == 1) {
        if (loading_tp_size_ <= 0 || loading_tp_rank_ < 0 || loading_tp_rank_ >= loading_tp_size_ ||
            in_features_ % size_t(loading_tp_size_))
            throw std::invalid_argument("GPTQ constructor row TP geometry is invalid");
        local_k = in_features_ / size_t(loading_tp_size_);
        if (local_k != metadata.k)
            throw std::invalid_argument("GPTQ constructor local K differs from packed checkpoint");
        if (metadata.group > 0) {
            if (local_k % size_t(metadata.group))
                throw std::invalid_argument("GPTQ constructor row TP shard crosses group boundary");
            const size_t local_groups = local_k / size_t(metadata.group);
            if (local_groups > size_t(INT32_MAX) ||
                size_t(loading_tp_rank_) > size_t(INT32_MAX) / local_groups)
                throw std::invalid_argument("GPTQ constructor group offset overflows int32");
            expected = size_t(loading_tp_rank_) * local_groups;
        }
    }
    if (local_k != metadata.k)
        throw std::invalid_argument("GPTQ constructor local K differs from packed checkpoint");
    const size_t tail = metadata.group == -1 ? 0 : (local_k / size_t(metadata.group) - 1);
    if (expected > size_t(INT32_MAX) || tail > size_t(INT32_MAX) - expected)
        throw std::invalid_argument("GPTQ constructor canonical group tail overflows int32");
    return int64_t(expected);
    }

std::shared_ptr<const infinilm::quantization::W4Execution> BaseLinear::prepare_w4_after_loading(
    infinilm::runtime::Runtime &runtime, bool retained_global_scale_table) {
    runtime.assertOwnerThread();
    if (!infinilm::quantization::is_w4(quantization_->get_quant_scheme()))
        throw std::logic_error("explicit W4 preparation requires AWQ/GPTQ layer");
    if (w4_prepared_ || w4_execution_) throw std::logic_error("W4 layer preparation is single-use");
    if (quantization_->get_quant_scheme() == infinilm::quantization::QuantScheme::AWQ_W4A16)
        validate_awq_metadata(runtime); // same pure predicate, never trust a prior ACK
    std::optional<int64_t> expected_non_desc_group_offset;
    if (quantization_->get_quant_scheme() == infinilm::quantization::QuantScheme::GPTQ_W4A16)
        expected_non_desc_group_offset = validate_gptq_metadata(runtime, retained_global_scale_table);
    auto candidate = quantization_->prepare_w4_after_loading(
        runtime, bound_params_, split_dim_, retained_global_scale_table, expected_non_desc_group_offset);
    w4_staged_identity_ = candidate;
    w4_prepared_ = true; // only after every validation, allocation, map copy and final drain succeeded
    return candidate;
}

void BaseLinear::publish_w4_after_loading(
    std::shared_ptr<const infinilm::quantization::W4Execution> candidate) noexcept {
    // Calling with null/mismatched/reused candidate is a violated commit contract,
    // never a silent Ready publication. This function does no allocation/work.
    if (!w4_prepared_ || w4_execution_ || !candidate || w4_staged_identity_.lock() != candidate) std::terminate();
    w4_execution_.swap(candidate);
    w4_staged_identity_.reset();
}

void BaseLinear::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_)
        throw std::invalid_argument("postload Runtime and Linear devices differ");
    if (infinilm::quantization::is_w4(quantization_->get_quant_scheme()))
        throw std::logic_error("W4 requires explicit Runtime stage/publish; recursive no-arg finalize is not supported");
    if (pre_transpose && runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("dense pre_transpose requires inactive graph state");
    if (typed_postload_started_)
        throw std::logic_error("typed Linear postload is single-use");
    // All old/new bindings stay live through completion proof on every failure.
    auto old_parameters = local_weights_;
    auto old_bound = bound_params_;
    auto old_quant = quantization_;
    auto params = bound_params_;
    decltype(local_weights_) prepared_parameters;
    std::shared_ptr<infinilm::quantization::BaseQuantization> new_quant;
    try {
        // A throwing quantization hook cannot be retried on a partially changed
        // object. Worker failure still owns the model-wide not-Ready decision.
        typed_postload_started_ = true;
        new_quant = quantization_->process_weights_after_loading(runtime, params, pre_transpose, split_dim_);
        if (!new_quant) return;
        prepared_parameters.reserve(params.size());
        for (const auto &[name, tensor] : params)
            prepared_parameters.emplace(name, infinilm::weights::Weight(tensor));
        // Same allocator, nonthrowing swaps: never publish a half-empty parameter map.
        local_weights_.swap(prepared_parameters);
        bound_params_.swap(params);
        quantization_.swap(new_quant);
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

// Backward compatible accessors

infinilm::runtime::Tensor BaseLinear::weight() const {
    auto it = local_weights_.find("weight");
    if (it != local_weights_.end()) {
        return it->second;
    }
    it = local_weights_.find("qweight");
    if (it != local_weights_.end()) {
        return it->second;
    }
    return infinilm::runtime::Tensor();
}

infinilm::runtime::Tensor BaseLinear::bias() const {
    auto it = local_weights_.find("bias");
    if (it != local_weights_.end()) {
        return it->second;
    }
    return infinilm::runtime::Tensor();
}

const infinilm::weights::Weight &BaseLinear::get_parameter_ref(const std::string &name) const {
    return local_weights_.at(name);
}

std::vector<infinilm::quantization::SplitParam> BaseLinear::split_params(
    const std::vector<infinilm::quantization::SplitInfo> &splits,
    int tp_rank, int tp_size, int tp_num_heads) const {
    if (w4_prepared_ || w4_execution_)
        throw std::logic_error("W4 fused raw slicing must precede explicit preparation");
    return quantization_->split_params(
        local_weights_, splits, split_dim_, tp_rank, tp_size, tp_num_heads);
}

} // namespace infinilm::nn
