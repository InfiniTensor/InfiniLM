#include "../weights/weights.hpp"
#include "infer_engine.hpp"
#include "kimi_engine_input_r145.hpp"
#include "runtime/runtime.hpp"
#include "../cache/kv_cache.hpp"
#include "../config/config_factory.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <cmath>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <unordered_set>

namespace infinilm::engine {
namespace {

// Checked value issuer only: no identities/leases/requests stored globally.
// Unique across live engines/resets in this native process; never wraps.
std::uint64_t issue_nsa_pool_epoch() {
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;) {
        if (value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("NSA native pool epoch exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
            return value;
    }
}


bool is_internal_moe_packed_weight(const std::string &key) {
    constexpr const char *suffixes[] = {
        ".mlp.experts.w13_weight", ".mlp.experts.w2_weight",
        ".mlp.experts.w1", ".mlp.experts.w2"};
    for (const auto *suffix : suffixes) {
        const auto length = std::char_traits<char>::length(suffix);
        if (key.size() >= length && key.compare(key.size() - length, length, suffix) == 0) {
            return true;
        }
    }
    return false;
}

std::string weight_key_error(const char *kind, std::vector<std::string> keys) {
    std::sort(keys.begin(), keys.end());
    std::string result = std::string(kind) + " key(s) in state_dict: ";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i) result += ", ";
        result += '"' + keys[i] + '"';
    }
    return result;
}

size_t max_length_from_offsets(
    const std::optional<infinilm::runtime::Tensor> &offsets,
    const char *name) {
    if (!offsets.has_value()) {
        return 0;
    }

    auto cpu_offsets = offsets.value();
    if (cpu_offsets->device().type() != infinilm::runtime::Device::Type::kCpu) {
        cpu_offsets = cpu_offsets->to(
            infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    }

    if (cpu_offsets->dtype() != infinilm::runtime::DataType::kInt32
        || cpu_offsets->shape().size() != 1
        || cpu_offsets->shape()[0] < 2) {
        throw std::invalid_argument(
            std::string(name) + " must be a one-dimensional int32 tensor with at least two entries");
    }

    const auto *values = reinterpret_cast<const int32_t *>(cpu_offsets->data());
    size_t max_length = 0;
    for (size_t i = 1; i < cpu_offsets->shape()[0]; ++i) {
        if (values[i] < values[i - 1]) {
            throw std::invalid_argument(std::string(name) + " must be nondecreasing");
        }
        max_length = std::max(
            max_length,
            static_cast<size_t>(values[i] - values[i - 1]));
    }
    return max_length;
}

std::optional<size_t> first_sequence_length(
    const std::optional<infinilm::runtime::Tensor> &lengths,
    const char *name) {
    if (!lengths.has_value()) {
        return std::nullopt;
    }

    auto cpu_lengths = lengths.value();
    if (cpu_lengths->device().type() != infinilm::runtime::Device::Type::kCpu) {
        cpu_lengths = cpu_lengths->to(
            infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    }

    if (cpu_lengths->dtype() != infinilm::runtime::DataType::kInt32
        || cpu_lengths->shape().size() != 1
        || cpu_lengths->shape()[0] == 0) {
        throw std::invalid_argument(
            std::string(name) + " must be a non-empty one-dimensional int32 tensor");
    }

    const auto value = reinterpret_cast<const int32_t *>(cpu_lengths->data())[0];
    if (value < 0) {
        throw std::invalid_argument(std::string(name) + " must contain nonnegative lengths");
    }
    return static_cast<size_t>(value);
}

} // namespace

//------------------------------------------------------
// Constructor
//------------------------------------------------------
InferEngine::InferEngine(
    const std::string &config_str,
    const distributed::DistConfig &distributed_config,
    infinilm::runtime::Device::Type device_type,
    const cache::CacheConfig *cache_config,
    bool enable_graph_compiling,
    backends::AttentionBackend attention_backend,
    std::optional<infinilm::runtime::DataType> kv_cache_dtype,
    bool use_mla,
    const std::string &weight_load_mode,
    bool pre_transpose)
    : communication_group_(distributed_config, device_type),
      attention_backend_(attention_backend),
      weight_load_mode_(weight_load_mode),
      use_mla_(use_mla) {
    if (weight_load_mode_ != "async" && weight_load_mode_ != "sync") {
        throw std::invalid_argument("weight_load_mode must be either 'async' or 'sync'");
    }
    if (attention_backend_ == backends::AttentionBackend::FLASH_ATTN) {
        // These backends register both canonical InfiniOps flash-attention
        // providers used by the paged prefill and decode paths below.
        if (device_type != infinilm::runtime::Device::Type::kNvidia
            && device_type != infinilm::runtime::Device::Type::kMetax
            && device_type != infinilm::runtime::Device::Type::kMoore
            && device_type != infinilm::runtime::Device::Type::kCambricon
            && device_type != infinilm::runtime::Device::Type::kIluvatar
            && device_type != infinilm::runtime::Device::Type::kHygon
            && device_type != infinilm::runtime::Device::Type::kAscend) {
            throw std::invalid_argument(
                "flash-attn is only available on NVIDIA, MetaX, Moore, Cambricon, Iluvatar, Hygon, and Ascend devices");
        }
        const auto *paged_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (paged_cache_config == nullptr) {
            throw std::invalid_argument(
                "flash-attn requires a paged KV cache configuration");
        }
        if (paged_cache_config->num_blocks() == 0) {
            throw std::invalid_argument(
                "flash-attn requires at least one paged KV cache block");
        }
        if (paged_cache_config->block_size() == 0
            || paged_cache_config->block_size() % 256 != 0) {
            throw std::invalid_argument(
                "flash-attn requires a nonzero paged KV cache block size divisible by 256");
        }
    }
    if (cache_config != nullptr) {
        cache_config_ = cache_config->unique_copy();
    }

    // Load model config if model_path is provided, model_path must be valid, and config.json exists
    this->model_config_ = infinilm::config::ConfigFactory::createConfig(config_str);
    const auto public_model_type = model_config_->get<std::string>("model_type");
    if (public_model_type == "mistral") {
        const auto *static_cache = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config_.get());
        const auto *paged_cache = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
        const bool static_layout = attention_backend_ == backends::AttentionBackend::STATIC_ATTN
            && static_cache && static_cache->max_cache_len() == 256;
        const bool paged_layout = attention_backend_ == backends::AttentionBackend::FLASH_ATTN
            && paged_cache && paged_cache->block_size() == 256;
        if (device_type != infinilm::runtime::Device::Type::kNvidia
            || distributed_config.tp_device_ids.size() != 1
            || distributed_config.pp_size != 1 || distributed_config.pp_stage != 0
            || distributed_config.moe_ep_backend != "disabled" || distributed_config.moe_ep_size != 1
            || enable_graph_compiling || pre_transpose || use_mla || kv_cache_dtype.has_value()
            || (!static_layout && !paged_layout))
            throw std::invalid_argument("Public Mistral requires NVIDIA TP1/PP1, disabled EP, eager static256 or flash/paged256 attention, pre_transpose=false and NONE KV quantization; MLA is unsupported");
    }
    if (public_model_type == "qwen3_moe" || public_model_type == "deepseek"
        || public_model_type == "mamba" || public_model_type == "ernie4_5_moe_vl") {
        if (device_type != infinilm::runtime::Device::Type::kNvidia
            || distributed_config.tp_device_ids.size() != 1
            || distributed_config.pp_size != 1 || distributed_config.pp_stage != 0
            || distributed_config.moe_ep_backend != "disabled" || distributed_config.moe_ep_size != 1
            || enable_graph_compiling || pre_transpose || use_mla || kv_cache_dtype.has_value()
            || attention_backend_ != backends::AttentionBackend::STATIC_ATTN
            || dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config_.get()) == nullptr) {
            throw std::invalid_argument(
                std::string("Public ") + (public_model_type == "deepseek" ? "Deepseek"
                    : public_model_type == "mamba" ? "Mamba"
                    : public_model_type == "ernie4_5_moe_vl" ? "Ernie" : "Qwen3MoE")
                + " requires NVIDIA TP1/PP1, disabled EP, static KV cache, "
                  "eager execution, pre_transpose=false and NONE KV quantization; MLA is unsupported");
        }
    }
    if (public_model_type == "ernie4_5_moe_vl"
        && dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config_.get())->max_cache_len() != 256)
        throw std::invalid_argument("Public Ernie text requires static cache length256");
    const auto quant_scheme = model_config_->get_quant_scheme();
    if (quant_scheme == quantization::QuantScheme::AWQ_W4A16
        || quant_scheme == quantization::QuantScheme::GPTQ_W4A16) {
        if (device_type != infinilm::runtime::Device::Type::kNvidia
            || distributed_config.tp_device_ids.size() != 1
            || distributed_config.pp_size != 1 || distributed_config.pp_stage != 0
            || distributed_config.moe_ep_backend != "disabled" || distributed_config.moe_ep_size != 1
            || enable_graph_compiling || pre_transpose || use_mla || kv_cache_dtype.has_value()
            || model_config_->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE)
            throw std::invalid_argument(
                std::string("Public ")
                + (quant_scheme == quantization::QuantScheme::AWQ_W4A16 ? "AWQ" : "GPTQ")
                + " requires NVIDIA TP1/PP1, disabled EP, eager execution, "
                  "pre_transpose=false and NONE KV quantization; MLA is unsupported");
    }
    if (attention_backend_ == backends::AttentionBackend::FLASH_ATTN) {
        const auto dtype = model_config_->get_dtype();
        if (dtype != infinilm::runtime::DataType::kFloat16
            && dtype != infinilm::runtime::DataType::kBFloat16) {
            throw std::invalid_argument(
                "flash-attn requires a float16 or bfloat16 model dtype");
        }
        const size_t head_dim = model_config_->get_head_dim();
        if (head_dim == 0 || head_dim > 256 || head_dim % 8 != 0) {
            throw std::invalid_argument(
                "flash-attn requires head_dim to be a positive multiple of 8 no greater than 256");
        }
        if (device_type == infinilm::runtime::Device::Type::kMoore
            && head_dim != 64
            && head_dim != 128) {
            throw std::invalid_argument(
                "flash-attn on Moore requires head_dim to be 64 or 128");
        }
        if (device_type == infinilm::runtime::Device::Type::kIluvatar
            && head_dim != 64
            && head_dim != 128) {
            throw std::invalid_argument(
                "flash-attn on Iluvatar requires head_dim to be 64 or 128");
        }
    }
    auto infinilm_config = std::make_shared<infinilm::global_state::InfinilmConfig>(
        attention_backend,
        this->model_config_,
        use_mla,
        distributed_config.moe_ep_backend,
        distributed_config.moe_ep_size,
        pre_transpose);

    // Only support offline int8 kv cache quantization in this version
    if (kv_cache_dtype.has_value()) {
        this->model_config_->set_kv_quant_scheme(kv_cache_dtype.value());
    }
    if (model_config_->get<std::string>("model_type") == "videonsa"
        && dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get()) != nullptr)
    {
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
        if (enable_graph_compiling || distributed_config.pp_size != 1 || use_mla
            || device_type != infinilm::runtime::Device::Type::kNvidia || !paged->block_size()
            || paged->block_size() % 64 || !paged->num_blocks()
            || paged->block_size() > 2147483647u || paged->num_blocks() > 2147483647u
            || model_config_->get_head_dim() != 128
            || (model_config_->get_dtype() != infinilm::runtime::DataType::kFloat16
                && model_config_->get_dtype() != infinilm::runtime::DataType::kBFloat16)
            || model_config_->get_quant_scheme() != quantization::QuantScheme::NONE
            || model_config_->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
            || (attention_backend_ != backends::AttentionBackend::PAGED_ATTN
                && attention_backend_ != backends::AttentionBackend::FLASH_ATTN))
            throw std::invalid_argument("NSA owner slice excludes graph/PP/MLA/quantization and unsupported cache geometry");
        nsa_pool_epoch_ = issue_nsa_pool_epoch();
    }
    // Create one RankWorker per rank
    int local_tp_size = communication_group_.tp_size();
    barrier_ = std::make_unique<RankBarrier>((size_t)local_tp_size);
    if (model_config_->get<std::string>("model_type") == "kimi_k3") {
        models::kimi_native::validate_cache(cache_config_.get());
        const auto& cfg=model_config_->get_config_json();
        if(enable_graph_compiling||pre_transpose||use_mla||kv_cache_dtype.has_value()||device_type!=infinilm::runtime::Device::Type::kHygon||local_tp_size!=8||distributed_config.pp_size!=4||distributed_config.pp_stage<0||distributed_config.pp_stage>=4||distributed_config.moe_ep_backend!="disabled"||distributed_config.moe_ep_size!=1||attention_backend_!=backends::AttentionBackend::FLASH_ATTN||!cfg.value("infinilm_text_only",false)||!cfg.value("infinilm_kimi_engine_adapter",false)||model_config_->get_dtype()!=infinilm::runtime::DataType::kBFloat16||model_config_->get_kv_quant_scheme()!=quantization::KVQuantAlgo::NONE||cfg.at("num_hidden_layers")!=93||cfg.at("hidden_size")!=7168||cfg.at("num_experts")!=896||cfg.at("num_experts_per_tok")!=16)
            throw std::invalid_argument("Kimi Engine requires actual93-layer Hygon TP8/PP4 eager BF16 text, model-owned MXFP4 routed weights, disabledEP and224/48 or288/64 pages/rows at256 width");
        kimi_native_lengths_.assign(models::kimi_native::request_capacity(cache_config_.get())+1,0);
    }
    workers_.reserve(local_tp_size);
    try {
        for (int r = 0; r < local_tp_size; ++r) {
        workers_.emplace_back(std::make_unique<RankWorker>(
            infinilm_config,
            communication_group_.get_rank_info(r),
            cache_config_ != nullptr ? cache_config_.get() : nullptr,
            barrier_.get(),
            enable_graph_compiling,
                attention_backend_, nsa_pool_epoch_));
    }

    for (auto &worker : workers_) {
        worker->wait_for_init();
        }
    } catch (...) {
        const auto error = std::current_exception();
        for (auto &worker : workers_) {
            worker->request_stop();
        }
        barrier_->abort(error);
        for (auto &worker : workers_) {
            worker->join();
        }
        std::rethrow_exception(error);
    }
    // Graphs must be compiled after weights are loaded and post-processed.
    // Quantized models may replace their linear implementations during
    // process_weights_after_loading(), so compiling here would capture stale
    // fallback operators.
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::load_param(const std::string &name, const infinilm::runtime::Tensor &param) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (weight_state_ != WeightState::Loading) {
        throw std::logic_error("weights can only be loaded while the engine is Loading");
    }
    weight_load_started_ = true;
    try {
        if (loaded_weight_keys_.count(name)) {
            throw std::runtime_error(weight_key_error("Duplicate", {name}));
        }
    for (auto &worker : workers_) {
        worker->load_param(name, param);
        }
        loaded_weight_keys_.insert(name);
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

void InferEngine::load_params(const std::unordered_map<std::string, infinilm::runtime::Tensor> &params, bool strict) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (weight_state_ != WeightState::Loading) {
        throw std::logic_error("weights can only be loaded while the engine is Loading");
    }
    if (!params.empty()) {
        weight_load_started_ = true;
    }
    try {
        std::vector<std::string> duplicates;
        for (const auto &[key, _] : params) {
            if (loaded_weight_keys_.count(key)) duplicates.push_back(key);
        }
        if (!duplicates.empty()) {
            throw std::runtime_error(weight_key_error("Duplicate", std::move(duplicates)));
        }
    if (workers_.size() <= 1 || weight_load_mode_ == "sync") {
        for (auto &worker : workers_) {
            worker->load_params(params, strict);
        }
        } else {
    std::vector<std::future<void>> futures;
    futures.reserve(workers_.size());
    for (auto &worker : workers_) {
                futures.emplace_back(std::async(std::launch::async, [worker_ptr = worker.get(), &params, strict] {
                    worker_ptr->load_params(params, strict);
        }));
    }
    for (auto &future : futures) {
        future.get();
            }
        }
        for (const auto &[key, _] : params) {
            loaded_weight_keys_.insert(key);
        }
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::process_weights_after_loading() {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (weight_state_ != WeightState::Loading) {
        throw std::logic_error("weights can only be finalized once from Loading");
    }
    weight_state_ = WeightState::Finalizing;
    try {
        const auto expected_keys = state_dict_keys();
        const std::unordered_set<std::string> expected(expected_keys.begin(), expected_keys.end());
        std::vector<std::string> missing;
        std::vector<std::string> unexpected;
        for (const auto &key : expected_keys) {
            if (!loaded_weight_keys_.count(key) && !is_internal_moe_packed_weight(key)) {
                missing.push_back(key);
            }
        }
        for (const auto &key : loaded_weight_keys_) {
            if (!expected.count(key)) unexpected.push_back(key);
        }
        if (!missing.empty() || !unexpected.empty()) {
            std::string message = "Error(s) in loading state_dict";
            if (!unexpected.empty()) message += "\n" + weight_key_error("Unexpected", std::move(unexpected));
            if (!missing.empty()) message += "\n" + weight_key_error("Missing", std::move(missing));
            throw std::runtime_error(message);
        }
        const auto scheme = model_config_->get_quant_scheme();
        if (scheme == infinilm::quantization::QuantScheme::AWQ_W4A16 ||
            scheme == infinilm::quantization::QuantScheme::GPTQ_W4A16) {
            if (model_config_->get<std::string>("model_type") != "qwen3")
                throw std::invalid_argument("W4 finalization supports only exact Qwen3 owner family");
            // Every worker rejects unsupported/mixed coverage before ANY stage.
            for (auto &worker : workers_) worker->validate_w4_weights();
            for (auto &worker : workers_) worker->prepare_w4_weights();
            // All prepare calls acknowledge successful owner sync/trim/seal.
            for (auto &worker : workers_) worker->commit_w4_weights();
        } else {
    for (auto &worker : workers_) {
        worker->process_weights_after_loading();
    }
        }
        compile_workers();
        weight_state_ = WeightState::Ready;
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

//------------------------------------------------------
// state_dict
//------------------------------------------------------
std::vector<std::unordered_map<std::string, infinilm::weights::Weight>> InferEngine::state_dict() {
    std::vector<std::unordered_map<std::string, infinilm::weights::Weight>> results;
    if (0 == workers_.size()) {
        throw std::runtime_error(" Model object not found. ");
    }

    for (auto &worker : workers_) {
        results.push_back(worker->state_dict());
    }
    return results;
}

std::vector<std::string> InferEngine::state_dict_keys() {
    if (0 == workers_.size()) {
        throw std::runtime_error(" Model object not found. ");
    }
    std::vector<std::string> ordered_keys;
    std::unordered_set<std::string> seen_keys;
    for (auto &worker : workers_) {
        for (const auto &key : worker->state_dict_keys()) {
            // Preserve first-seen worker order while removing duplicate TP keys.
            if (seen_keys.emplace(key).second) {
                ordered_keys.push_back(key);
            }
        }
    }
    return ordered_keys;
}

//------------------------------------------------------
// forward
//------------------------------------------------------
infinilm::InfinilmModel::Input
InferEngine::Input::to_model_input(
    infinilm::runtime::Device device,
    ForwardContext &forward_context,
    bool snapshot_static_sequence_lengths,
    bool preserve_target_hidden_device,
    size_t logical_kv_upper_bound,
    std::shared_ptr<const nsa::Proof> nsa_owner) const {

    forward_context.nsa_owner = std::move(nsa_owner);
    auto &runtime = forward_context.require_runtime();
    if (device.type() != infinilm::runtime::Device::Type::kCpu && runtime.device() != device)
        throw std::invalid_argument("input transfer Runtime and target device differ");
    auto to_device = [&](const std::optional<infinilm::runtime::Tensor> &t)
        -> std::optional<infinilm::runtime::Tensor> {
        if (!t.has_value() || t.value()->device() == device) return t;
        if (t.value()->device().type() == infinilm::runtime::Device::Type::kCpu
            && device.type() != infinilm::runtime::Device::Type::kCpu) {
            auto result = infinilm::runtime::Tensor::empty(runtime, t.value()->shape(), t.value()->dtype());
            result->copy_from(runtime, t.value());
            return result;
        }
        auto source = t.value()->allocation_owner();
        if (!source || source->device() != t.value()->device())
            source = infinilm::runtime::context::resolveRuntime(t.value()->device());
        return device.type() == infinilm::runtime::Device::Type::kCpu
            ? t.value()->to(*source, device) : t.value()->to(*source, runtime);
    };
    auto to_device_vec = [&](const std::optional<std::vector<infinilm::runtime::Tensor>> &vec)
        -> std::optional<std::vector<infinilm::runtime::Tensor>> {
        if (!vec.has_value()) {
            return vec;
        }
        std::vector<infinilm::runtime::Tensor> result;
        result.reserve(vec->size());
        for (const auto &t : vec.value()) {
            result.push_back(to_device(t).value());
        }
        return result;
    };

    const bool is_prefill = input_ids.has_value()
                         && total_sequence_lengths.has_value()
                         && input_ids.value()->numel()
                                != total_sequence_lengths.value()->numel();
    const size_t max_query_length = is_prefill ? max_length_from_offsets(input_offsets, "input_offsets") : 0;
    const size_t max_sequence_length = is_prefill ? max_length_from_offsets(cu_seqlens, "cu_seqlens") : 0;

    std::optional<size_t> first_past_sequence_length;
    std::optional<size_t> first_total_sequence_length;
    if (snapshot_static_sequence_lengths) {
        first_past_sequence_length = first_sequence_length(
            past_sequence_lengths, "past_sequence_lengths");
        first_total_sequence_length = first_sequence_length(
            total_sequence_lengths, "total_sequence_lengths");
    }

    infinilm::InfinilmModel::Input input = {
        to_device(input_ids), // @todo: on device in the future
        to_device(position_ids),
        to_device(past_sequence_lengths), // @todo: on device in the future
        to_device(total_sequence_lengths),
        to_device(input_offsets),
        to_device(cu_seqlens),
        to_device(block_tables),
        to_device(slot_mapping),
        to_device(mamba_init_state_indices),
        to_device(mamba_final_state_indices),
        to_device_vec(pixel_values),
        to_device_vec(image_bound),
        to_device_vec(tgt_sizes),
        to_device_vec(image_grid_thw),
        image_req_ids,
        visual_token_ranges,
        preserve_target_hidden_device
            ? target_hidden_states
            : to_device(target_hidden_states),
        sample_all_positions};
    input.logical_kv_upper_bound = logical_kv_upper_bound;

    auto &attn_metadata = forward_context.attn_metadata;
    attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping,
        max_query_length,
        max_sequence_length};
    attn_metadata.first_past_sequence_length = first_past_sequence_length;
    attn_metadata.first_total_sequence_length = first_total_sequence_length;
    attn_metadata.logical_kv_upper_bound = logical_kv_upper_bound;

    forward_context.mamba_metadata = {
        input.input_offsets,
        input.mamba_init_state_indices,
        input.mamba_final_state_indices};

    forward_context.mm_metadata = {
        image_req_ids,
        visual_token_ranges};

    return input;
}

InferEngine::Output InferEngine::forward(const InferEngine::Input &provided) {
    auto input=provided;
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);

    if (weight_state_ == WeightState::Failed
        || weight_state_ == WeightState::Finalizing
        || (weight_state_ == WeightState::Loading && weight_load_started_)) {
        throw std::logic_error("cannot execute an incomplete or failed weight load");
    }
    if (input.random_values.has_value()) {
        if (input.sample_all_positions || !input.input_offsets.has_value()
            || input.input_offsets.value()->ndim() != 1
            || input.input_offsets.value()->size(0) < 2
            || input.random_values->size() != input.input_offsets.value()->size(0) - 1) {
            throw std::invalid_argument(
                "random_values requires one value per request and last-token sampling");
        }
        for (float value : *input.random_values) {
            if (!std::isfinite(value) || value < 0.0f || value >= 1.0f) {
                throw std::invalid_argument("random_values must be finite values in [0, 1)");
            }
        }
    }
    if (input.sampling_rows.has_value()) {
        if (input.sample_all_positions || !input.input_offsets.has_value()
            || input.input_offsets.value()->ndim() != 1
            || input.input_offsets.value()->size(0) < 2
            || input.sampling_rows->size() != input.input_offsets.value()->size(0) - 1) {
            throw std::invalid_argument(
                "sampling_rows requires one value per request and last-token sampling");
        }
        for (const auto &[temperature, top_p, top_k] : *input.sampling_rows) {
            if (!std::isfinite(temperature) || temperature < 0.0f
                || !std::isfinite(top_p) || top_p < 0.0f || top_p > 1.0f
                || top_k <= 0) {
                throw std::invalid_argument("sampling_rows contains invalid parameters");
            }
        }
    }
    std::vector<models::kimi_native::Request> kimi_plan;
    if(model_config_->get<std::string>("model_type")=="kimi_k3")kimi_plan=models::kimi_native::freeze_input(input,kimi_native_lengths_,model_config_->get<size_t>("vocab_size"));
    // Trigger each worker to run inference
    for (auto &worker : workers_) {
        worker->run(input);
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }

    if(!kimi_plan.empty())models::kimi_native::commit(kimi_native_lengths_,kimi_plan);
    auto output = workers_[0]->get_output();
    // Completed worker output remains a mutable borrowed view, not a snapshot.
    const auto completed_view = [](infinilm::runtime::Tensor tensor) {
        if (!tensor) return tensor;
        return infinilm::runtime::Tensor::strided_from_blob(
            tensor->data(), tensor->shape(), tensor->strides(),
            tensor->dtype(), tensor->device(), tensor->shared_from_this());
    };
    output.output_ids = completed_view(output.output_ids);
    output.logits = completed_view(output.logits);
    output.hidden_states = completed_view(output.hidden_states);
    return output;

}

void InferEngine::compile() {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (model_config_->get<std::string>("model_type") == "kimi_k3")
        throw std::runtime_error("KimiK3 indexed KDA is unsupported: native Core adapter/provider unavailable; provider and state restoration pending");
    if (nsa_pool_epoch_) throw std::invalid_argument("NSA owner slice excludes COMPILE before state mutation");
    if (weight_state_ == WeightState::Loading) {
        return;
    }
    if (weight_state_ != WeightState::Ready) {
        throw std::logic_error("cannot compile an incomplete or failed weight load");
    }
    try {
        compile_workers();
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

void InferEngine::compile_workers() {
    if (model_config_->get<std::string>("model_type") == "kimi_k3") return; // Unsupported K3 must not enter graph preparation.
    if (nsa_pool_epoch_) return; // weight finalization/reset must not enter COMPILE
    for (auto &worker : workers_) {
        worker->compile();
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }
}

//------------------------------------------------------
// Destructor
//------------------------------------------------------
InferEngine::~InferEngine() {
    // Close all workers
    for (auto &worker : workers_) {
        worker->request_stop();
    }
    if (barrier_ != nullptr) {
        barrier_->abort(nullptr);
    }
    for (auto &worker : workers_) {
        worker->join();
    }
    // Workers retain models and recorded calls that can refer to communicators.
    // Destroy them while communication_group_ is still alive; member destruction
    // alone would run in the opposite order of these declarations.
    workers_.clear();
}

const distributed::DistConfig &InferEngine::get_dist_config() const {
    return communication_group_.get_dist_config();
}

//------------------------------------------------------
// reset_cache (overloaded with CacheConfig)
//------------------------------------------------------
void InferEngine::reset_cache(const cache::CacheConfig *new_config) {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if(model_config_->get<std::string>("model_type")=="kimi_k3")models::kimi_native::validate_cache(new_config);
    if (new_config == nullptr) {
        throw std::invalid_argument("reset_cache requires a cache configuration");
    }
    if (weight_state_ != WeightState::Loading
        && weight_state_ != WeightState::Ready) {
        throw std::logic_error("cannot reset cache during an incomplete or failed weight load");
    }
    if (nsa_pool_epoch_) {
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(new_config);
        if (!paged || !paged->block_size() || paged->block_size() % 64 || !paged->num_blocks()
            || paged->block_size() > 2147483647u || paged->num_blocks() > 2147483647u
            || (attention_backend_ == backends::AttentionBackend::FLASH_ATTN && paged->block_size() % 256))
            throw std::invalid_argument("NSA reset requires supported paged geometry before owner mutation");
    }
    const auto replacement_epoch = nsa_pool_epoch_ ? issue_nsa_pool_epoch() : 0;
    try {
    for (auto &worker : workers_) {
            worker->reset_cache(new_config, replacement_epoch);
    }
    for (auto &worker : workers_) {
        worker->wait();
    }
    cache_config_ = new_config->unique_copy();
        nsa_pool_epoch_ = replacement_epoch; // publish only after all workers succeeded
    if(model_config_->get<std::string>("model_type")=="kimi_k3")kimi_native_lengths_.assign(models::kimi_native::request_capacity(cache_config_.get())+1,0);
        if (weight_state_ == WeightState::Loading) {
            return;
        }
        compile_workers();
    } catch (...) {
        weight_state_ = WeightState::Failed;
        throw;
    }
}

std::uint64_t InferEngine::get_nsa_pool_epoch() {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (!nsa_pool_epoch_ || workers_.empty() || weight_state_ == WeightState::Failed)
        throw std::logic_error("NSA pool query requires a live VideoNSA engine");
    for (const auto &worker : workers_)
        if (worker->get_nsa_pool_epoch() != nsa_pool_epoch_)
            throw std::logic_error("NSA engine/worker pool epoch mismatch");
    return nsa_pool_epoch_;
}

std::vector<std::vector<infinilm::runtime::Tensor>> InferEngine::get_kv_cache() {
    std::lock_guard<std::mutex> submission_lock(submission_mutex_);
    if (model_config_->get<std::string>("model_type") == "kimi_k3")
        throw std::runtime_error("KimiK3 indexed KDA is unsupported: native Core adapter/provider unavailable; provider and state restoration pending");
    if (nsa_pool_epoch_) throw std::invalid_argument("NSA borrowed KV writers are outside owner slice");
    std::vector<std::vector<infinilm::runtime::Tensor>> kv_cache_list;
    if (workers_.empty()) {
        throw std::runtime_error("InferEngine::get_cache_vec: no workers");
    }

    kv_cache_list.reserve(workers_.size());
    for (auto &worker : workers_) {
        kv_cache_list.push_back(std::move(worker->get_kv_cache()));
    }

    for (auto &worker : workers_) {
        worker->wait();
    }

    return kv_cache_list;
}

} // namespace infinilm::engine
