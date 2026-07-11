#include "infer_engine.hpp"
#include "../config/config_factory.hpp"
#include "spdlog/spdlog.h"
#include <future>
#include <stdexcept>
#include <unordered_set>

namespace infinilm::engine {

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
    const std::string &weight_load_mode)
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
        distributed_config.moe_ep_size);

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
InferEngine::Input::to_model_input(infinicore::Device device) const {

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
        to_device(target_hidden_states)};

    infinilm::global_state::get_forward_context().attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping};

    infinilm::global_state::get_forward_context().mamba_metadata = {
        input.input_offsets,
        input.mamba_init_state_indices,
        input.mamba_final_state_indices};

    global_state::get_forward_context().mm_metadata = {
        image_req_ids,
        visual_token_ranges};

    return input;
}

InferEngine::Output InferEngine::forward(const InferEngine::Input &input) {
    Input local_input = input;
    if (last_output_ready_event_ && local_input.input_ids.has_value() &&
        local_input.input_ids.value()->device().getType() != infinicore::Device::Type::CPU) {
        // Decode feeds the previous sampled GPU tensor back as input. Keep the
        // worker stream ordered without forcing a host synchronization.
        local_input.wait_event = last_output_ready_event_;
    }

    // Trigger each worker to run inference
    for (auto &worker : workers_) {
        worker->run(local_input);
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }

    auto output = workers_[0]->get_output();
    last_output_ids_ = output.output_ids;
    last_output_ready_event_ = output.ready_event;
    return output;
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
    // Keep graph-registered allreduce buffers alive with the compiled CUDA graph.
    // Prefill/decode backend selection should be explicit instead of unregistering here.
}

//------------------------------------------------------
// Destructor
//------------------------------------------------------
InferEngine::~InferEngine() {
    close();
}

void InferEngine::close() {
    if (closed_) {
        return;
    }
    closed_ = true;
    sync_last_output();
    last_output_ready_event_.reset();
    last_saved_output_event_.reset();
    last_output_ids_.reset();
    request_output_refs_.clear();
    for (auto &worker : workers_) {
        worker->request_close();
    }
    for (auto &worker : workers_) {
        worker->join();
    }
    workers_.clear();
    barrier_.reset();
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
    reset_request_state();
    this->compile();
}

void InferEngine::reset_request_state() {
    sync_last_output();
    last_output_ready_event_.reset();
    last_output_ids_.reset();
    last_saved_output_event_.reset();
    request_output_refs_.clear();
}

void InferEngine::sync_last_output() {
    if (last_output_ready_event_) {
        last_output_ready_event_->synchronize();
    }
    if (last_saved_output_event_) {
        last_saved_output_event_->synchronize();
    }
    request_output_refs_.clear();
}

void InferEngine::copy_last_output_to(infinicore::Tensor dst) {
    if (!last_output_ids_) {
        throw std::runtime_error("No previous output tensor is available to copy");
    }
    if (!dst) {
        throw std::runtime_error("Destination output tensor is empty");
    }
    if (dst->shape() != last_output_ids_->shape()) {
        throw std::runtime_error(
            "Cannot copy output with different shape. Src: " + last_output_ids_->info() +
            " Dst: " + dst->info());
    }
    if (!(dst->device() == last_output_ids_->device())) {
        throw std::runtime_error(
            "Destination output tensor must be on the same device as the sampled token. Src: " +
            last_output_ids_->info() + " Dst: " + dst->info());
    }

    infinicore::context::setDevice(dst->device());
    if (last_saved_output_event_ && last_saved_output_event_->is_recorded() && last_saved_output_event_->query()) {
        request_output_refs_.clear();
    }
    if (last_output_ready_event_) {
        infinicore::context::streamWaitEvent(
            infinicore::context::getStream(), last_output_ready_event_->get());
    }
    dst->copy_from(last_output_ids_);
    request_output_refs_.push_back(last_output_ids_);
    if (!last_saved_output_event_ || !(last_saved_output_event_->device() == dst->device())) {
        last_saved_output_event_ = std::make_shared<infinicore::DeviceEvent>(dst->device());
    }
    last_saved_output_event_->record(infinicore::context::getStream());
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
