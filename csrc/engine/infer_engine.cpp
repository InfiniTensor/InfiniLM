#include "infer_engine.hpp"
#include "../config/config_factory.hpp"
#include "infinicore/ops/distributed/broadcast.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cstdint>
#include <future>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

namespace infinilm::engine {
namespace {

size_t max_length_from_offsets(
    const std::optional<infinicore::Tensor> &offsets,
    const char *name) {
    if (!offsets.has_value()) {
        return 0;
    }

    auto cpu_offsets = offsets.value();
    if (cpu_offsets->device().getType() != infinicore::Device::Type::CPU) {
        cpu_offsets = cpu_offsets->to(infinicore::Device::cpu());
        infinicore::context::syncStream();
    }

    if (cpu_offsets->dtype() != infinicore::DataType::I32
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

} // namespace

//------------------------------------------------------
// Constructor
//------------------------------------------------------
InferEngine::InferEngine(
    const std::string &config_str,
    const distributed::DistConfig &distributed_config,
    infinicore::Device::Type device_type,
    const cache::CacheConfig *cache_config,
    bool enable_graph_compiling,
    backends::AttentionBackend attention_backend,
    std::optional<infinicore::DataType> kv_cache_dtype,
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
    if (cache_config != nullptr) {
        cache_config_ = cache_config->unique_copy();
    }

    // Load model config if model_path is provided, model_path must be valid, and config.json exists
    this->model_config_ = infinilm::config::ConfigFactory::createConfig(config_str);
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
    // Create one RankWorker per rank
    int world_size = communication_group_.get_world_size();
    barrier_ = std::make_unique<RankBarrier>((size_t)world_size);
    workers_.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
        workers_.emplace_back(std::make_unique<RankWorker>(
            infinilm_config,
            communication_group_.get_rank_info(r),
            cache_config_ != nullptr ? cache_config_.get() : nullptr,
            barrier_.get(),
            enable_graph_compiling,
            attention_backend_));
    }

    for (auto &worker : workers_) {
        worker->wait_for_init();
    }
    // Graphs must be compiled after weights are loaded and post-processed.
    // Quantized models may replace their linear implementations during
    // process_weights_after_loading(), so compiling here would capture stale
    // fallback operators.
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::load_param(const std::string &name, const infinicore::Tensor &param) {
    // Load the parameter on all workers
    for (auto &worker : workers_) {
        worker->load_param(name, param);
    }
}

void InferEngine::load_params(const std::unordered_map<std::string, infinicore::Tensor> &params, bool strict) {
    if (workers_.size() <= 1 || weight_load_mode_ == "sync") {
        for (auto &worker : workers_) {
            worker->load_params(params, strict);
        }
        return;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(workers_.size());
    for (auto &worker : workers_) {
        futures.emplace_back(std::async(std::launch::async, [&worker, &params, strict] {
            worker->load_params(params, strict);
        }));
    }
    for (auto &future : futures) {
        future.get();
    }
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::process_weights_after_loading() {
    // Process the weights after loading on all workers
    for (auto &worker : workers_) {
        worker->process_weights_after_loading();
    }
    weights_finalized_ = true;
    this->compile();
}

//------------------------------------------------------
// state_dict
//------------------------------------------------------
std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> InferEngine::state_dict() {
    std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> results;
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
InferEngine::Input::to_model_input(infinicore::Device device, bool for_graph) const {

    auto to_device = [&](const std::optional<infinicore::Tensor> &t)
        -> std::optional<infinicore::Tensor> {
        return t.has_value() ? t.value()->to(device) : t;
    };
    auto to_device_vec = [&](const std::optional<std::vector<infinicore::Tensor>> &vec)
        -> std::optional<std::vector<infinicore::Tensor>> {
        if (!vec.has_value()) {
            return vec;
        }
        std::vector<infinicore::Tensor> result;
        result.reserve(vec->size());
        for (const auto &t : vec.value()) {
            result.push_back(t->to(device));
        }
        return result;
    };

    const bool is_prefill = input_ids.has_value()
                         && total_sequence_lengths.has_value()
                         && input_ids.value()->numel()
                                != total_sequence_lengths.value()->numel();
    const size_t max_query_length = is_prefill ? max_length_from_offsets(input_offsets, "input_offsets") : 0;
    const size_t max_sequence_length = is_prefill ? max_length_from_offsets(cu_seqlens, "cu_seqlens") : 0;

    if (token_state_indices.has_value()) {
        auto host_indices = [](const std::optional<infinicore::Tensor> &value, size_t count) {
            if (!value || value.value()->device().getType() != infinicore::Device::Type::CPU
                || value.value()->dtype() != infinicore::DataType::I32
                || value.value()->shape() != std::vector<size_t>{count}
                || !value.value()->is_contiguous()) {
                throw std::runtime_error("Token state checkpoints require contiguous CPU int32 indices.");
            }
            return reinterpret_cast<const int32_t *>(value.value()->data());
        };
        if (!input_ids || input_ids.value()->ndim() != 2 || input_ids.value()->size(0) != 1
            || !input_offsets || input_offsets.value()->numel() < 2 || target_hidden_states) {
            throw std::runtime_error("Token checkpoints require packed target-model requests.");
        }
        const size_t tokens = input_ids.value()->size(1);
        const size_t requests = input_offsets.value()->numel() - 1;
        const auto *destinations = host_indices(token_state_indices, tokens);
        const auto *initial = host_indices(mamba_init_state_indices, requests);
        const auto *final = host_indices(mamba_final_state_indices, requests);
        const auto *offsets = host_indices(input_offsets, requests + 1);
        if (offsets[0] != 0 || offsets[requests] != static_cast<int32_t>(tokens)) {
            throw std::runtime_error("Token checkpoint offsets must cover all packed tokens.");
        }
        std::unordered_set<int32_t> used;
        for (size_t r = 0; r < requests; ++r) {
            if (offsets[r] < 0 || offsets[r + 1] <= offsets[r]
                || static_cast<size_t>(offsets[r + 1]) > tokens) {
                throw std::runtime_error("Token checkpoint offsets must be increasing and within the packed input.");
            }
            const auto length = offsets[r + 1] - offsets[r];
            if (length > 8 || destinations[offsets[r + 1] - 1] != final[r]) {
                throw std::runtime_error("Each checkpoint request needs 1..8 tokens and a matching final row.");
            }
            if (initial[r] != 0 && !used.insert(initial[r]).second) {
                throw std::runtime_error("Checkpoint requests must own distinct initial state rows.");
            }
            used.insert(initial[r]);
        }
        for (size_t t = 0; t < tokens; ++t) {
            if (destinations[t] <= 0 || !used.insert(destinations[t]).second) {
                throw std::runtime_error("Token checkpoints must use distinct nonzero destination rows.");
            }
        }
        const auto &context = global_state::get_forward_context();
        for (const auto *states : {&context.conv_state_vec, &context.ssm_state_vec}) {
            for (const auto &state : *states) {
                if (state) {
                    for (auto index : used) {
                        if (index < 0 || static_cast<size_t>(index) >= state->size(0)) {
                            throw std::runtime_error("Token checkpoint exceeds the allocated state pool.");
                        }
                    }
                }
            }
        }
    }

    const auto transfer_device = for_graph ? global_state::get_tensor_model_parallel_rank_info().device : device;
    auto distribute = [&](const std::optional<infinicore::Tensor> &value, int source_rank) {
        if (!value || source_rank < 0 || transfer_device.getType() == infinicore::Device::Type::CPU) {
            return value;
        }
        const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
        if (rank_info.tp_size == 1) {
            // Same-device inputs also need compact storage for model kernels.
            return std::optional<infinicore::Tensor>{value.value()->contiguous()};
        }
        const auto &source = value.value();
        auto local = rank_info.tp_rank == source_rank
                       ? source->contiguous()
                       : infinicore::Tensor::empty(source->shape(), source->dtype(), transfer_device);
        // `Tensor::to` does not transfer between distinct GPUs. Use the existing
        // TP communicator for both hidden states and device-resident candidates.
        infinicore::op::distributed::broadcast_(local, local, source_rank, rank_info.comm);
        return std::optional<infinicore::Tensor>{local};
    };
    auto local_target_hidden = distribute(target_hidden_states, target_hidden_source_rank);
    auto local_ids = distribute(input_ids, input_source_rank);

    // MACA maps a registered user pointer to only one node. Serialize H2D
    // copies so TP ranks never access the same host registration concurrently.
    // Collectives above must stay outside this lock so all ranks can enter.
    static std::mutex maca_host_copy_mutex;
    const bool serialize_host_copy
        = device.getType() == infinicore::Device::Type::METAX;
    std::unique_lock<std::mutex> maca_host_copy_lock;
    if (serialize_host_copy) {
        maca_host_copy_lock = std::unique_lock<std::mutex>(maca_host_copy_mutex);
    }

    infinilm::InfinilmModel::Input input = {
        for_graph ? local_ids : to_device(local_ids),
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
        for_graph ? local_target_hidden : to_device(local_target_hidden),
        sample_all_positions,
        to_device(token_state_indices),
        top_k == 1 && !return_logits};

    if (serialize_host_copy) {
        infinicore::context::syncStream();
    }

    infinilm::global_state::get_forward_context().attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping,
        max_query_length,
        max_sequence_length};

    // Expand only attention rows, giving every speculative query its causal
    // length and its own request's page table. GDN/Conv retain packed offsets.
    const bool short_draft = target_hidden_states && input_offsets
                          && max_query_length <= 8;
    if ((token_state_indices || short_draft) && is_prefill && input.block_tables
        && device.getType() == infinicore::Device::Type::NVIDIA
        && max_sequence_length > max_query_length) {
        auto host_offsets = input_offsets.value()->to(infinicore::Device::cpu())->contiguous();
        auto host_lengths = total_sequence_lengths.value()->to(infinicore::Device::cpu())->contiguous();
        if (input_offsets.value()->device().getType() != infinicore::Device::Type::CPU
            || total_sequence_lengths.value()->device().getType() != infinicore::Device::Type::CPU) {
            infinicore::context::syncStream();
        }
        const size_t requests = host_offsets->numel() - 1;
        if (host_lengths->dtype() != infinicore::DataType::I32
            || host_lengths->shape() != std::vector<size_t>{requests}
            || input.block_tables.value()->ndim() != 2
            || input.block_tables.value()->size(0) != requests) {
            throw std::invalid_argument("Short verification needs one KV length and page-table row per request.");
        }
        const auto *offsets = reinterpret_cast<const int32_t *>(host_offsets->data());
        const auto *totals = reinterpret_cast<const int32_t *>(host_lengths->data());
        const size_t tokens = input_ids.value()->numel();
        if (offsets[0] != 0 || offsets[requests] != static_cast<int32_t>(tokens)) {
            throw std::invalid_argument("Short verification offsets must cover all query tokens.");
        }
        std::vector<int32_t> lengths(tokens);
        auto &metadata = global_state::get_forward_context().attn_metadata;
        const auto &tables = input.block_tables.value();
        auto expanded_tables = infinicore::Tensor::empty({tokens, tables->size(1)}, tables->dtype(), device);
        for (size_t r = 0; r < requests; ++r) {
            const auto count = offsets[r + 1] - offsets[r];
            if (count <= 0 || totals[r] < count) {
                throw std::invalid_argument("Short verification requires nonempty queries within each KV length.");
            }
            for (int32_t t = 0; t < count; ++t) {
                lengths[offsets[r] + t] = totals[r] - count + t + 1;
            }
            auto repeated = tables->narrow({{0, r, 1}})->as_strided({static_cast<size_t>(count), tables->size(1)}, {0, tables->stride(1)});
            expanded_tables->narrow({{0, static_cast<size_t>(offsets[r]), static_cast<size_t>(count)}})->copy_from(repeated);
        }
        metadata.verification_sequence_lengths = infinicore::Tensor::empty(
            {lengths.size()}, infinicore::DataType::I32, device);
        infinicore::context::memcpyH2D(metadata.verification_sequence_lengths.value()->data(),
                                       lengths.data(), lengths.size() * sizeof(int32_t), false);
        metadata.verification_block_tables = std::move(expanded_tables);
    }

    infinilm::global_state::get_forward_context().mamba_metadata = {
        input.input_offsets,
        input.mamba_init_state_indices,
        input.mamba_final_state_indices,
        input.token_state_indices};
    if (token_state_indices) {
        const auto *offsets = reinterpret_cast<const int32_t *>(input_offsets.value()->data());
        global_state::get_forward_context().mamba_metadata.checkpoint_offsets.assign(
            offsets, offsets + input_offsets.value()->numel());
    }

    global_state::get_forward_context().mm_metadata = {
        image_req_ids,
        visual_token_ranges};

    return input;
}

InferEngine::Output InferEngine::forward(const InferEngine::Input &input) {
    auto local_input = input;
    auto source_rank = [&](const std::optional<infinicore::Tensor> &value) {
        if (!value || value.value()->device().getType() == infinicore::Device::Type::CPU) {
            return -1;
        }
        for (int rank = 0; rank < communication_group_.get_world_size(); ++rank) {
            if (communication_group_.get_rank_info(rank).device == value.value()->device()) {
                return rank;
            }
        }
        throw std::invalid_argument("Device inputs must belong to the target TP group.");
    };
    local_input.target_hidden_source_rank = source_rank(input.target_hidden_states);
    local_input.input_source_rank = get_dist_config().pp_size == 1 ? source_rank(input.input_ids) : -1;
    if (input.verify_draft) {
        const bool valid_shape = input.input_ids && input.input_ids.value()->ndim() == 2
                              && input.input_ids.value()->size(0) == 1
                              && input.input_ids.value()->size(1) >= 2
                              && input.input_ids.value()->size(1) <= 5
                              && input.input_offsets && input.input_offsets.value()->numel() == 2;
        if (input.top_k != 1 || !input.sample_all_positions || !input.token_state_indices
            || !valid_shape || get_dist_config().pp_size != 1 || input.return_device_tokens) {
            throw std::invalid_argument("Device MTP acceptance requires one greedy Q=2..5 request, checkpoints, PP1 and host results.");
        }
    }
    if (input.return_device_tokens && get_dist_config().pp_size != 1) {
        throw std::invalid_argument("Device token output currently requires PP1.");
    }
    // Trigger each worker to run inference
    for (auto &worker : workers_) {
        worker->run(local_input);
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }

    return workers_[0]->get_output();
}

void InferEngine::compile() {
    if (!weights_finalized_) {
        return;
    }
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
        worker->close();
    }
}

const distributed::DistConfig &InferEngine::get_dist_config() const {
    return communication_group_.get_dist_config();
}

//------------------------------------------------------
// reset_cache (overloaded with CacheConfig)
//------------------------------------------------------
void InferEngine::reset_cache(const cache::CacheConfig *new_config) {
    for (auto &worker : workers_) {
        worker->reset_cache(new_config);
    }
    for (auto &worker : workers_) {
        worker->wait();
    }
    cache_config_ = new_config->unique_copy();
    this->compile();
}

std::vector<std::vector<std::vector<infinicore::Tensor>>> InferEngine::get_hybrid_states() {
    std::vector<std::vector<std::vector<infinicore::Tensor>>> result;
    for (auto &worker : workers_) {
        worker->wait();
        result.push_back(worker->get_hybrid_states());
    }
    return result;
}

std::vector<std::vector<infinicore::Tensor>> InferEngine::get_kv_cache() {
    std::vector<std::vector<infinicore::Tensor>> kv_cache_list;
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
