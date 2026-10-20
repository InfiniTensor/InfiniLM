#include "../../weights/weights.hpp"
#include "runtime/utils.hpp"
#include "../../layers/linear/dense.hpp"
#include "mamba_for_causal_lm.hpp"
#include "../../execution/submit.hpp"
#include "../models_registry.hpp"
#include "runtime/runtime.hpp"
#ifndef INFINILM_MAMBA_SELECTIVE_SCAN_OPS_UNAVAILABLE
#include "base/mamba_selective_scan.h"
#endif
#include <stdexcept>
#include <exception>
#include <cstring>
#include <type_traits>
#include <limits>
#include <cmath>

namespace infinilm::models::mamba {

namespace {
template<class Owner>
void safe_metadata_d2h(Owner &owner, void *dst, const void *src, size_t bytes) {
    try { owner.memcpyD2H(dst, src, bytes); }
    catch (...) {
        try { owner.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::optional<std::vector<int64_t>> read_metadata(
    infinilm::runtime::Runtime &runtime, const std::optional<infinilm::runtime::Tensor> &value) {
    if (!value) return std::nullopt;
    const auto &tensor = *value;
    if (!tensor || tensor->ndim() != 1 || !tensor->is_contiguous() ||
        (tensor->dtype() != infinilm::runtime::DataType::kInt32 && tensor->dtype() != infinilm::runtime::DataType::kInt64)) {
        throw std::invalid_argument("Mamba metadata requires contiguous rank-one signed I32/I64");
    }
    if (tensor->device() != runtime.device() && tensor->device().type() != infinilm::runtime::Device::Type::kCpu) {
        throw std::invalid_argument("Mamba metadata belongs to a different Runtime device");
    }
    std::vector<int64_t> result(tensor->numel());
    auto read = [&](auto *typed) {
        using T = std::remove_pointer_t<decltype(typed)>;
        std::vector<T> values(result.size());
        if (!values.empty()) {
            if (tensor->device().type() == infinilm::runtime::Device::Type::kCpu) {
                std::memcpy(values.data(), tensor->data(), values.size() * sizeof(T));
            } else {
                safe_metadata_d2h(runtime, values.data(), tensor->data(), values.size() * sizeof(T));
            }
        }
        for (size_t i = 0; i < result.size(); ++i) result[i] = values[i];
    };
    if (tensor->dtype() == infinilm::runtime::DataType::kInt32) read(static_cast<int32_t *>(nullptr));
    else read(static_cast<int64_t *>(nullptr));
    return result;
}

void validate_pools(infinilm::runtime::Runtime &runtime,
                    const global_state::ForwardContext &context, size_t layers,
                    size_t channels, size_t states, size_t history,
                    infinilm::runtime::DataType dtype) {
    if (context.kv_cache_vec.size() != layers * 2) {
        throw std::invalid_argument("Mamba requires paired state pools for every layer");
    }
    for (size_t i = 0; i < layers; ++i) {
        const auto &conv = context.kv_cache_vec[2 * i];
        const auto &scan = context.kv_cache_vec[2 * i + 1];
        if (!conv || !scan || conv->device() != runtime.device() || scan->device() != runtime.device() ||
            conv->ndim() != 3 || scan->ndim() != 3 || !conv->is_contiguous() || !scan->is_contiguous() ||
            conv->dtype() != dtype || scan->dtype() != infinilm::runtime::DataType::kFloat32 ||
            conv->size(1) != channels || conv->size(2) != history ||
            scan->size(1) != channels || scan->size(2) != states ||
            conv->size(0) != scan->size(0) ||
            conv->size(0) != context.kv_cache_vec[0]->size(0)) {
            throw std::invalid_argument("Mamba paired state pools have incompatible layout/device/dtype");
        }
    }
    // Reject aliasing between any paired/layer pools before the first commit.
    // All accepted pools are contiguous, so a byte-span comparison is exact.
    for (size_t i = 0; i < context.kv_cache_vec.size(); ++i) {
        const auto &left = context.kv_cache_vec[i];
        const auto begin = reinterpret_cast<uintptr_t>(left->data());
        const auto bytes = left->nbytes();
        if (bytes && (!begin || bytes > std::numeric_limits<uintptr_t>::max() - begin)) {
            throw std::invalid_argument("Mamba state pool address is invalid");
        }
        for (size_t j = 0; j < i; ++j) {
            const auto &right = context.kv_cache_vec[j];
            const auto other = reinterpret_cast<uintptr_t>(right->data());
            if (bytes && right->nbytes() && begin < other + right->nbytes() && other < begin + bytes) {
                throw std::invalid_argument("Mamba state pools must not overlap");
            }
        }
    }
}

void copy_state_row(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor destination,
                     const infinilm::runtime::Tensor &source) {
    // Validated equal-layout contiguous rows: preserve even nonfinite state
    // bits, and avoid adding a dtype conversion or new operator dependency.
    if (source->nbytes()) runtime.memcpyD2D(destination->data(), source->data(), source->nbytes());
}
} // namespace

std::shared_ptr<infinilm::config::ModelConfig>
create_mamba_model_config(std::shared_ptr<infinilm::config::ModelConfig> config) {
    const std::string model_type = config->get<std::string>("model_type");
    if (model_type != "mamba") {
        throw std::runtime_error("mamba config creator called for non-mamba model");
    }
    const size_t limit = std::numeric_limits<size_t>::max();
    const auto product = [limit](size_t a, size_t b) {
        if (b && a > limit / b) throw std::length_error("Mamba construction size overflow");
        return a * b;
    };
    const auto positive = [&config](const char *key) {
        if (config->get<int64_t>(key) <= 0)
            throw std::invalid_argument(std::string("Mamba constructor requires positive ") + key);
        return config->get<size_t>(key);
    };
    // Original Mamba normalization reads these effective operands as int.
    // Reject narrowing before that handler, leaving its defaults/formulas intact.
    const auto &json = config->get_config_json();
    for (const char *key : {"hidden_size", "d_model", "num_hidden_layers", "n_layer", "state_size", "conv_kernel", "expand"}) {
        if (!json.contains(key)
            || (std::string(key) == "d_model" && json.contains("hidden_size"))
            || (std::string(key) == "n_layer" && json.contains("num_hidden_layers"))
            || (std::string(key) == "expand" && json.contains("intermediate_size"))) continue;
        const auto value = config->get<int64_t>(key);
        if (value <= 0 || value > std::numeric_limits<int>::max())
            throw std::invalid_argument(std::string("Mamba requires a positive int-representable ") + key);
    }
    const auto default_hidden = config->get_or<int64_t>("hidden_size", config->get_or<int64_t>("d_model", 2048));
    if (default_hidden <= 0 || static_cast<uint64_t>(default_hidden) > limit - 15)
        throw std::invalid_argument("Mamba requires valid hidden_size before default arithmetic");
    if (!config->get_config_json().contains("intermediate_size")) {
        const auto expand = config->get_or<int64_t>("expand", 2);
        if (expand <= 0) throw std::invalid_argument("Mamba requires positive expansion");
        (void)product(static_cast<size_t>(default_hidden), static_cast<size_t>(expand));
    }
    auto &j = config->get_config_json();
    j["hidden_size"] = j.value("hidden_size", j.value("d_model", 2048));
    j["num_hidden_layers"] = j.value("num_hidden_layers", j.value("n_layer", 48));
    j["intermediate_size"] = j.value("intermediate_size", j["hidden_size"].get<size_t>() * j.value("expand", 2));
    j["layer_norm_epsilon"] = j.value("layer_norm_epsilon", 1e-5);
    j["rms_norm_eps"] = j.value("rms_norm_eps", j["layer_norm_epsilon"]);
    j["state_size"] = j.value("state_size", 16);
    j["conv_kernel"] = j.value("conv_kernel", 4);
    if (!j.contains("time_step_rank")
        || (j["time_step_rank"].is_string() && j["time_step_rank"] == "auto")) {
        j["time_step_rank"] = (j["hidden_size"].get<size_t>() + 15) / 16;
    } else if (!j["time_step_rank"].is_number_integer()) {
        throw std::invalid_argument("Mamba time_step_rank requires auto or a positive integer");
    }
    if (j.contains("hidden_act") && (!j["hidden_act"].is_string() || j["hidden_act"] != "silu")) {
        throw std::invalid_argument("Mamba requires hidden_act=silu");
    }
    j["hidden_act"] = "silu";
    for (const char *key : {"residual_in_fp32", "tie_word_embeddings"}) {
        if (j.contains(key) && !j[key].is_boolean()) {
            throw std::invalid_argument(std::string("Mamba requires boolean ") + key);
        }
        if (!j.contains(key)) j[key] = true;
    }
    j["use_bias"] = j.value("use_bias", false);
    j["use_conv_bias"] = j.value("use_conv_bias", true);
    j["max_position_embeddings"] = j.value("max_position_embeddings", 8192);
    // Mamba has no attention/RoPE fields. These are its existing allocation expressions.
    const size_t hidden = positive("hidden_size"), vocab = positive("vocab_size"), layers = positive("num_hidden_layers");
    const size_t mid = positive("intermediate_size"), states = positive("state_size"), time_rank = positive("time_step_rank");
    const size_t kernel = positive("conv_kernel"), twice_states = product(states, 2);
    if (time_rank > limit - twice_states) throw std::length_error("Mamba packed state projection size overflow");
    const auto dtype = config->get_dtype();
    if (dtype != infinilm::runtime::DataType::kFloat32 && dtype != infinilm::runtime::DataType::kFloat16 && dtype != infinilm::runtime::DataType::kBFloat16)
        throw std::invalid_argument("Mamba requires floating model dtype");
    const size_t bytes = infinilm::runtime::dsize(dtype);
    (void)product(product(vocab, hidden), bytes);
    (void)product(product(hidden, product(mid, 2)), bytes);
    (void)product(product(mid, time_rank + twice_states), bytes);
    (void)product(product(time_rank, mid), bytes);
    (void)product(product(mid, hidden), bytes);
    (void)product(product(mid, kernel), bytes);
    (void)product(product(mid, states), bytes);
    (void)product(mid, bytes);
    (void)product(layers, sizeof(std::shared_ptr<infinilm::weights::ModelWeights>));
    const double epsilon = config->get<double>("layer_norm_epsilon");
    if (!std::isfinite(epsilon) || epsilon <= 0) throw std::invalid_argument("Mamba requires finite positive norm epsilon");
    return config;
}

MambaMixer::MambaMixer(infinilm::runtime::Runtime &runtime,
                       std::shared_ptr<infinilm::config::ModelConfig> config,
                       size_t layer_idx, const engine::distributed::RankInfo &rank_info,
                       backends::AttentionBackend attention_backend,
                       const layers::moe::EPConfig &ep_config)
    : layer_idx_(layer_idx) {
    models::validate_runtime_construction(config, runtime, rank_info, attention_backend, ep_config);
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    intermediate_size_ = config->get<size_t>("intermediate_size");
    state_size_ = config->get<size_t>("state_size");
    time_step_rank_ = config->get<size_t>("time_step_rank");
    conv_kernel_ = config->get<size_t>("conv_kernel");
    const bool use_bias = config->get_or<bool>("use_bias", false);
    const bool use_conv_bias = config->get_or<bool>("use_conv_bias", true);

    try {
        in_proj_ = this->bind_child("in_proj", std::make_shared<std::remove_reference_t<decltype(*in_proj_ )>>(runtime, hidden_size, intermediate_size_ * 2, use_bias, dtype));
        x_proj_ = this->bind_child("x_proj", std::make_shared<std::remove_reference_t<decltype(*x_proj_ )>>(runtime, intermediate_size_, time_step_rank_ + state_size_ * 2, false, dtype));
        dt_proj_ = this->bind_child("dt_proj", std::make_shared<std::remove_reference_t<decltype(*dt_proj_ )>>(runtime, time_step_rank_, intermediate_size_, true, dtype));
        out_proj_ = this->bind_child("out_proj", std::make_shared<std::remove_reference_t<decltype(*out_proj_ )>>(runtime, intermediate_size_, hidden_size, use_bias, dtype));
        conv1d_weight_ = infinilm::weights::Weight(runtime, {intermediate_size_, 1, conv_kernel_}, dtype); this->bind_weight("conv1d_weight", conv1d_weight_);
    if (use_conv_bias) {
            conv1d_bias_ = infinilm::weights::Weight(runtime, {intermediate_size_}, dtype); this->bind_weight("conv1d_bias", conv1d_bias_);
    }
        A_log_ = infinilm::weights::Weight(runtime, {intermediate_size_, state_size_}, dtype); this->bind_weight("A_log", A_log_);
        D_ = infinilm::weights::Weight(runtime, {intermediate_size_}, dtype); this->bind_weight("D", D_);
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    } catch (...) {
        // BODY catch: all Parameters are still members here.
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor MambaMixer::forward(const infinilm::runtime::Tensor &hidden_states,
                                      const global_state::ForwardContext &forward_context,
                                      const state_routing::Plan *route) const {
    if (!forward_context.runtime) {
        throw std::runtime_error("Mamba: explicit forward requires a Runtime");
    }
    auto &runtime = *forward_context.runtime;
    runtime.assertOwnerThread();
    if (route && runtime.isGraphRecording()) {
        throw std::invalid_argument("Mamba indexed routing requires eager execution");
    }
    auto x_mut = const_cast<infinilm::runtime::Tensor &>(hidden_states);
    auto projected = in_proj_->forward(runtime, x_mut);
    auto x = projected->narrow({{2, 0, intermediate_size_}})->contiguous(runtime);
    auto gate = projected->narrow({{2, intermediate_size_, intermediate_size_}})->contiguous(runtime);

    const auto &cache_vec = forward_context.kv_cache_vec;
    const size_t conv_idx = layer_idx_ * 2;
    const size_t ssm_idx = conv_idx + 1;
    if (cache_vec.size() <= ssm_idx) {
        throw std::runtime_error("MambaMixer: Mamba state cache is not allocated");
    }
    auto conv_pool = cache_vec[conv_idx];
    auto ssm_pool = cache_vec[ssm_idx];
    auto conv_state = conv_pool;
    auto ssm_state = ssm_pool;
    if (route) {
        // Snapshot BOTH pools before either stage commits: cycles and repeated
        // initial slots must read the start-of-layer state, not earlier writes.
        conv_state = infinilm::runtime::Tensor::empty(runtime, {route->requests, intermediate_size_, conv_kernel_ - 1}, conv_pool->dtype());
        ssm_state = infinilm::runtime::Tensor::empty(runtime, {route->requests, intermediate_size_, state_size_}, infinilm::runtime::DataType::kFloat32);
        state_routing::gather(*route, [&](size_t request, size_t slot) {
            copy_state_row(runtime, conv_state->narrow({{0, request, 1}}), conv_pool->narrow({{0, slot, 1}}));
            copy_state_row(runtime, ssm_state->narrow({{0, request, 1}}), ssm_pool->narrow({{0, slot, 1}}));
        });
    } else {
        const auto requests = hidden_states->size(0);
        if (!conv_pool || !ssm_pool || conv_pool->size(0) < requests || ssm_pool->size(0) < requests) {
            throw std::invalid_argument("Mamba prefix state pool too small");
        }
        conv_state = conv_pool->narrow({{0, 0, requests}});
        ssm_state = ssm_pool->narrow({{0, 0, requests}});
    }

    auto conv_out = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    const auto &conv_weight = conv1d_weight_.tensor();
    const auto &conv_bias_tensor = conv1d_bias_.tensor();
    const std::optional<infinilm::runtime::Tensor> conv_bias = conv_bias_tensor
        ? std::optional<infinilm::runtime::Tensor>{conv_bias_tensor} : std::nullopt;
    const std::optional<infinilm::runtime::Tensor> no_conv_tensor;
    if (!conv_state || !conv_weight)
        throw std::invalid_argument("causal conv required tensor missing");
    infinilm::execution::submit<infini::ops::CausalConv1d>(
        runtime, infinilm::execution::RecordingPolicy::HostOnly,
        infini::ops::CausalConv1d::DefaultConfig(runtime.device().type()),
        x, conv_weight, conv_state, conv_bias, no_conv_tensor, no_conv_tensor,
        no_conv_tensor, conv_out, std::optional<infinilm::runtime::Tensor>{conv_state});
    INFINILM_TRACE_OP(CausalConv1d, conv_out);
    if (route) {
        // Preserve the existing eager boundary: convolution state is committed
        // before activation/projection/scan; a later failure is not rollback.
        state_routing::commit(*route, [&](size_t slot, size_t request) {
            copy_state_row(runtime, conv_pool->narrow({{0, slot, 1}}), conv_state->narrow({{0, request, 1}}));
        });
    }
    auto activated = infinilm::runtime::Tensor::empty(runtime, conv_out->shape(), conv_out->dtype());
    infinilm::execution::submit<infini::ops::Silu>(
        runtime, infini::ops::Silu::DefaultConfig(runtime.device().type()), conv_out, activated);
    conv_out = activated;

    auto ssm_params_in = conv_out;
    auto ssm_params = x_proj_->forward(runtime, ssm_params_in);
    auto time_step = ssm_params->narrow({{2, 0, time_step_rank_}})->contiguous(runtime);
    auto b = ssm_params->narrow({{2, time_step_rank_, state_size_}})->contiguous(runtime);
    auto c = ssm_params->narrow({{2, time_step_rank_ + state_size_, state_size_}})->contiguous(runtime);
    // HF applies the learned dt bias inside scan, immediately before softplus.
    // Keep the existing Parameter binding and its postload weight orientation.
    auto dt = infinilm::layers::linear::dense_projection(
        runtime, time_step, dt_proj_->weight(), std::nullopt, dt_proj_->alpha(),
        std::nullopt, dt_projection_prepacked_);
    auto scan = infinilm::runtime::Tensor::empty(runtime, conv_out->shape(), conv_out->dtype());
#if defined(INFINILM_MAMBA_SELECTIVE_SCAN_OPS_UNAVAILABLE)
    throw std::runtime_error("MambaSelectiveScan public API is unavailable in the installed SDK");
#else
    infinilm::execution::submit<infini::ops::MambaSelectiveScan>(
        runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
        infini::ops::MambaSelectiveScan::DefaultConfig(runtime.device().type()),
        scan, conv_out, dt, b, c, A_log_.tensor(), D_.tensor(), gate, dt_proj_->bias(), ssm_state);
#endif
    INFINILM_TRACE_OP(MambaSelectiveScan, scan);
    if (route) {
        state_routing::commit(*route, [&](size_t slot, size_t request) {
            copy_state_row(runtime, ssm_pool->narrow({{0, slot, 1}}), ssm_state->narrow({{0, request, 1}}));
        });
    }
    auto scan_mut = scan;
    return out_proj_->forward(runtime, scan_mut);
}

MambaBlock::MambaBlock(infinilm::runtime::Runtime &runtime,
                       std::shared_ptr<infinilm::config::ModelConfig> config,
                       size_t layer_idx, const engine::distributed::RankInfo &rank_info,
                       backends::AttentionBackend attention_backend,
                       const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(config, runtime, rank_info, attention_backend, ep_config);
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const double eps = config->get<double>("layer_norm_epsilon");
    try {
        norm_ = this->bind_child("norm", std::make_shared<std::remove_reference_t<decltype(*norm_ )>>(runtime, hidden_size, eps, dtype));
        mixer_ = this->bind_child("mixer", std::make_shared<std::remove_reference_t<decltype(*mixer_ )>>(runtime, config, layer_idx, rank_info, attention_backend, ep_config));
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor MambaBlock::forward(const infinilm::runtime::Tensor &hidden_states,
                                      const global_state::ForwardContext &forward_context,
                                      const state_routing::Plan *route) const {
    if (!forward_context.runtime) {
        throw std::runtime_error("Mamba: explicit forward requires a Runtime");
    }
    auto &runtime = *forward_context.runtime;
    runtime.assertOwnerThread();
    auto residual = hidden_states;
    // In the public all-FP32 profile this already honors residual_in_fp32;
    // reduced-dtype residual conversion is a separate provider qualification.
    auto x = norm_->forward(runtime, hidden_states);
    x = mixer_->forward(x, forward_context, route);
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

MambaModel::MambaModel(infinilm::runtime::Runtime &runtime,
                       std::shared_ptr<infinilm::config::ModelConfig> config,
                       const engine::distributed::RankInfo &rank_info,
                       backends::AttentionBackend attention_backend,
                       const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(config, runtime, rank_info, attention_backend, ep_config);
    const auto &dtype = config->get_dtype();
    const size_t vocab_size = config->get<size_t>("vocab_size");
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = config->get<size_t>("num_hidden_layers");
    const double eps = config->get<double>("layer_norm_epsilon");
    routing_channels_ = config->get<size_t>("intermediate_size");
    routing_states_ = config->get<size_t>("state_size");
    routing_history_ = config->get<size_t>("conv_kernel") - 1;
    routing_dtype_ = dtype;
    try {
        embeddings_ = this->bind_child("embeddings", std::make_shared<std::remove_reference_t<decltype(*embeddings_ )>>(runtime, vocab_size, hidden_size, std::nullopt, dtype));
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(this->bind_child("layers." + std::to_string(i), std::make_shared<MambaBlock>(runtime, config, i, rank_info, attention_backend, ep_config)));
    }
        norm_f_ = this->bind_child("norm_f", std::make_shared<std::remove_reference_t<decltype(*norm_f_ )>>(runtime, hidden_size, eps, dtype));
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor MambaModel::forward(const infinilm::InfinilmModel::Input &input,
                                      const global_state::ForwardContext &forward_context) const {
    if (!forward_context.runtime) {
        throw std::runtime_error("Mamba: explicit forward requires a Runtime");
    }
    auto &runtime = *forward_context.runtime;
    runtime.assertOwnerThread();
    auto input_ids = input.input_ids.value();
    if (input_ids->shape().size() == 1) {
        input_ids = input_ids->view({1, input_ids->shape()[0]});
    }
    if (input_ids->ndim() != 2) {
        throw std::invalid_argument("Mamba input_ids must be rank one or two");
    }
    validate_pools(runtime, forward_context, layers_.size(), routing_channels_, routing_states_, routing_history_, routing_dtype_);
    const auto &metadata = forward_context.mamba_metadata;
    const bool has_metadata = metadata.input_offsets || metadata.init_state_indices || metadata.final_state_indices;
    std::optional<state_routing::Plan> plan;
    if (has_metadata) {
        // The local host mapping is eager-only. Host-read indices must never be
        // baked into a recorded graph whose metadata can change on replay.
        if (runtime.isGraphRecording()) {
            throw std::invalid_argument("Mamba indexed/packed routing requires eager execution");
        }
        plan = state_routing::make_plan(input_ids->size(0), input_ids->size(1),
            forward_context.kv_cache_vec.empty() ? 0 : forward_context.kv_cache_vec[0]->size(0),
            read_metadata(runtime, metadata.input_offsets),
            read_metadata(runtime, metadata.init_state_indices),
            read_metadata(runtime, metadata.final_state_indices));
    }
    const auto requests = plan ? plan->requests : input_ids->size(0);
    if (!forward_context.kv_cache_vec.empty() && requests > forward_context.kv_cache_vec[0]->size(0)) {
        throw std::invalid_argument("Mamba prefix state pool too small");
    }
    auto hidden_states = embeddings_->forward(runtime, input_ids);
    if (plan && plan->reshape_packed) {
        hidden_states = hidden_states->view({plan->requests, plan->length, hidden_states->size(2)});
    }
    const auto *route = plan && plan->indexed ? &*plan : nullptr;
    for (const auto &layer : layers_) {
        hidden_states = layer->forward(hidden_states, forward_context, route);
    }
    hidden_states = norm_f_->forward(runtime, hidden_states);
    if (plan && plan->reshape_packed) {
        hidden_states = hidden_states->view({1, input_ids->size(1), hidden_states->size(2)});
    }
    return hidden_states;
}

MambaForCausalLM::MambaForCausalLM(infinilm::runtime::Runtime &runtime,
                                   std::shared_ptr<infinilm::config::ModelConfig> config,
                                   const engine::distributed::RankInfo &rank_info,
                                   backends::AttentionBackend attention_backend,
                                   const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(config, runtime, rank_info, attention_backend, ep_config);
    model_config_ = config;
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t vocab_size = config->get<size_t>("vocab_size");
    try {
        backbone_ = this->bind_child("backbone", std::make_shared<std::remove_reference_t<decltype(*backbone_ )>>(runtime, config, rank_info, attention_backend, ep_config));
        lm_head_ = this->bind_child("lm_head", std::make_shared<std::remove_reference_t<decltype(*lm_head_ )>>(runtime, hidden_size, vocab_size, false, dtype));
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

InfinilmModel::Output MambaForCausalLM::forward(const InfinilmModel::Input &input,
                                             global_state::ForwardContext &forward_context) const {
    if (!forward_context.runtime) {
        throw std::runtime_error("Mamba: explicit forward requires a Runtime");
    }
    auto &runtime = *forward_context.runtime;
    runtime.assertOwnerThread();
    auto hidden_states = backbone_->forward(input, forward_context);
    auto logits = lm_head_->forward(runtime, hidden_states);
    return {logits, hidden_states};
}

void MambaForCausalLM::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank_info) {
    (void)attention_backend; // Recurrent state sizing depends on CacheConfig, not attention layout.
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr, forward_context, attention_backend, rank_info);
        return;
    }
    if (!forward_context.runtime) {
        throw std::runtime_error("Mamba: cache allocation requires a Runtime");
    }
    auto &runtime = *forward_context.runtime;
    cache::validate_cache_runtime(runtime, rank_info);
    cache_config_ = cache_config->unique_copy();
    auto &cache_vec = forward_context.kv_cache_vec;
    cache_vec.clear();
    const size_t num_layers = model_config_->get<size_t>("num_hidden_layers");
    const size_t intermediate = model_config_->get<size_t>("intermediate_size");
    const size_t state_size = model_config_->get<size_t>("state_size");
    const size_t conv_kernel = model_config_->get<size_t>("conv_kernel");
    size_t max_batch_size = 1;
    if (auto static_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        max_batch_size = static_config->max_batch_size();
    } else if (auto paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        max_batch_size = paged_config->num_blocks();
    }
    const auto &dtype = model_config_->get_dtype();
    cache_vec.reserve(num_layers * 2);
    infinilm::runtime::Tensor conv_state, ssm_state;
    try {
    for (size_t i = 0; i < num_layers; ++i) {
            conv_state = infinilm::runtime::Tensor::empty(runtime, {max_batch_size, intermediate, conv_kernel - 1}, dtype);
            if (conv_state->is_contiguous() && !runtime.isGraphRecording()) {
                if (!infini::ops::Fill::valid_output(infinilm::execution::detail::view(conv_state)))
                    throw std::invalid_argument("zero-fill output layout or storage is invalid");
                runtime.setDeviceMemoryAsync(conv_state->data(), 0, conv_state->nbytes(), runtime.stream());
            } else {
                infinilm::execution::submit<infini::ops::Fill>(
                    runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
                    conv_state, 0.0, conv_state);
            }
            INFINILM_TRACE_OP(Zeros, conv_state);
            cache_vec.push_back(conv_state);
            ssm_state = infinilm::runtime::Tensor::empty(runtime, {max_batch_size, intermediate, state_size}, infinilm::runtime::DataType::kFloat32);
            if (ssm_state->is_contiguous() && !runtime.isGraphRecording()) {
                if (!infini::ops::Fill::valid_output(infinilm::execution::detail::view(ssm_state)))
                    throw std::invalid_argument("zero-fill output layout or storage is invalid");
                runtime.setDeviceMemoryAsync(ssm_state->data(), 0, ssm_state->nbytes(), runtime.stream());
            } else {
                infinilm::execution::submit<infini::ops::Fill>(
                    runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
                    ssm_state, 0.0, ssm_state);
            }
            INFINILM_TRACE_OP(Zeros, ssm_state);
            cache_vec.push_back(ssm_state);
        }
    } catch (...) {
        // FC retains earlier entries; the pending tensors retain this iteration.
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

} // namespace infinilm::models::mamba

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    mamba,
    infinilm::models::mamba::MambaForCausalLM,
    infinilm::models::mamba::create_mamba_model_config);
} // namespace
