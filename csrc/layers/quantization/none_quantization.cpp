#include "none_quantization.hpp"
#include "../../weights/weights.hpp"
#include "../linear/dense.hpp"
#include "execution/submit.hpp"
#include "runtime/runtime.hpp"
#include <base/trans_matmul_weight_nz.h>
#include <exception>
#include <optional>
#include <stdexcept>

namespace infinilm::quantization {

NoneQuantization::NoneQuantization() : NoneQuantization(nlohmann::json()) {}

std::vector<ParamDescriptor> NoneQuantization::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinilm::runtime::DataType &dtype,
    bool bias) const {

    if (output_dtype_ && *output_dtype_ != dtype
        && (dtype != infinilm::runtime::DataType::kBFloat16
            || *output_dtype_ != infinilm::runtime::DataType::kFloat32 || bias || split_dim >= 0)) {
        throw std::invalid_argument("mixed dense output requires unsharded BF16 weight, FP32 output and no bias");
    }
    std::vector<ParamDescriptor> descs;
    const bool tensor_parallel = split_dim >= 0;
    descs.push_back({"weight", {out_features, in_features}, dtype, tensor_parallel ? split_dim : 0, tensor_parallel ? tp_rank : 0, tensor_parallel ? tp_size : 1});
    if (bias) {
        const bool column_parallel = split_dim == 0;
        descs.push_back(
            {"bias", {out_features}, dtype, 0, column_parallel ? tp_rank : 0, column_parallel ? tp_size : 1});
    }
    return descs;
}

infinilm::runtime::Tensor NoneQuantization::forward(
    infinilm::runtime::Runtime &runtime,
    const ParamsMap &params,
    const infinilm::runtime::Tensor &input,
    bool has_bias,
    float alpha) const {

    auto weight = params.at("weight");

    std::optional<infinilm::runtime::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    // Validate before either contiguous operation can allocate/copy/submit.
    const auto resolved = output_dtype_.value_or(input ? input->dtype() : infinilm::runtime::DataType::kFloat32);
    infinilm::layers::linear::validate_projection(
        runtime, input, weight, bias_opt, resolved, weight_layout_ != WeightLayout::None);
    auto input_contiguous = input->contiguous(runtime);

    if (weight_layout_ == WeightLayout::AscendNz) {
        auto output = infinilm::layers::linear::dense_projection_nz(
            runtime, input_contiguous, weight, bias_opt, alpha, output_dtype_);
        return narrow_nz_output(runtime, output, weight);
    }
    if (weight_layout_ == WeightLayout::PackedNd) {
        return infinilm::layers::linear::dense_projection(
            runtime, input_contiguous, static_cast<infinilm::runtime::Tensor>(weight),
            bias_opt, alpha, output_dtype_, true);
    }
    return infinilm::layers::linear::dense_projection(runtime, input_contiguous, static_cast<infinilm::runtime::Tensor>(weight->contiguous(runtime)), bias_opt, alpha, output_dtype_, false);
}

std::vector<SplitParam> NoneQuantization::split_params(
    const std::unordered_map<std::string, infinilm::weights::Weight> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {

    std::vector<SplitParam> result;
    auto weight_it = params.find("weight");
    auto bias_it = params.find("bias");
    int weight_narrow_dim = narrow_dim;
    if (weight_layout_ != WeightLayout::None) {
        if (narrow_dim != 0 && narrow_dim != 1) {
            throw std::invalid_argument(
                "prepacked linear weight split dimension must be 0 or 1");
        }
        weight_narrow_dim = 1 - narrow_dim;
    }

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinilm::weights::Weight(
                              weight_it->second->narrow({{static_cast<size_t>(weight_narrow_dim), s.start, s.size}}),
                              weight_narrow_dim, tp_rank, tp_size, s.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinilm::weights::Weight(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

std::shared_ptr<BaseQuantization> NoneQuantization::process_weights_after_loading(
    infinilm::runtime::Runtime &runtime, ParamsMap &params, bool pre_transpose,
    int split_dim) const {
    if (!pre_transpose) {
        return nullptr; // No TLS, allocation, submission or success fence.
    }
    runtime.assertOwnerThread();
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("dense pre_transpose requires inactive graph state");
    }
    const auto weight_it = params.find("weight");
    auto self = std::const_pointer_cast<BaseQuantization>(shared_from_this());
    infinilm::runtime::Tensor source, permuted, pending, padded, nz_weight;
    if (weight_it != params.end()) {
        source = weight_it->second;
        if (!source || source->device() != runtime.device()) {
            throw std::invalid_argument("pre_transpose Runtime and weight devices differ");
        }
        try {
            permuted = source->permute({1, 0});
            pending = infinilm::runtime::Tensor::empty(runtime, permuted->shape(), permuted->dtype());
            pending->copy_from(runtime, permuted);
            weight_it->second = pending;
            weight_layout_ = WeightLayout::PackedNd;

            const auto dtype = pending->dtype();
            const bool dtype_eligible = dtype == infinilm::runtime::DataType::kFloat16
                                     || dtype == infinilm::runtime::DataType::kBFloat16;
            const bool output_eligible = !output_dtype_ || *output_dtype_ == dtype;
            const size_t packed_k = pending->size(0);
            const size_t packed_n = pending->size(1);
            const bool pad_eligible = packed_n % 16 != 0 && split_dim < 0
                                   && params.find("bias") == params.end();
            if (runtime.device().type() == infinilm::runtime::Device::Type::kAscend
                && dtype_eligible && output_eligible && pending->ndim() == 2
                && packed_k % 16 == 0 && (packed_n % 16 == 0 || pad_eligible)) {
                const infinilm::runtime::Tensor *conversion_source = &pending;
                if (packed_n % 16 != 0) {
                    const size_t padded_n = (packed_n + 15) / 16 * 16;
                    padded = infinilm::runtime::Tensor::empty(runtime, {packed_k, padded_n}, dtype);
                    padded->narrow({{1, 0, packed_n}})->copy_from(runtime, pending);
                    conversion_source = &padded;
                }
                nz_weight = infinilm::runtime::Tensor::empty(
                    runtime, (*conversion_source)->shape(), dtype);
                infinilm::execution::submit<infini::ops::TransMatmulWeightNz>(
                    runtime, infinilm::execution::RecordingPolicy::HostOnly,
                    infini::ops::TransMatmulWeightNz::DefaultConfig(runtime.device().type()),
                    *conversion_source, nz_weight);
                weight_it->second = nz_weight;
                weight_layout_ = WeightLayout::AscendNz;
                nz_output_width_ = packed_n;
            }
        } catch (...) {
            const auto failure = std::current_exception();
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
            std::rethrow_exception(failure);
        }
    }
    return self;
}

infinilm::runtime::Tensor NoneQuantization::narrow_nz_output(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &output,
    const infinilm::runtime::Tensor &weight) const {
    if (nz_output_width_ == 0 || nz_output_width_ == weight->size(1)) {
        return output;
    }
    return output->narrow({{output->ndim() - 1, 0, nz_output_width_}})->contiguous(runtime);
}

} // namespace infinilm::quantization
