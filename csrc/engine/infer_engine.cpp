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

    const auto &dist_config = communication_group_.get_dist_config();
    const size_t pp_size = dist_config.pp_device_ids.size();
    if (pp_size > 1) {
        if (dist_config.tp_device_ids.size() != 1) {
            throw std::invalid_argument("Pipeline parallel MVP requires tensor parallel size == 1");
        }
        if (device_type != infinicore::Device::Type::NVIDIA) {
            throw std::invalid_argument(
                "Pipeline parallel MVP currently supports NVIDIA devices only");
        }
        if (dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get()) == nullptr) {
            throw std::invalid_argument(
                "Pipeline parallel MVP requires paged KV cache");
        }
        if (attention_backend != backends::AttentionBackend::PAGED_ATTN) {
            throw std::invalid_argument(
                "Pipeline parallel MVP requires paged-attn attention backend");
        }
        if (use_mla) {
            throw std::invalid_argument(
                "Pipeline parallel MVP does not support MLA");
        }
        if (enable_graph_compiling) {
            throw std::invalid_argument("Pipeline parallel MVP does not support graph compiling");
        }
        if (dist_config.moe_ep_backend != "disabled" || dist_config.moe_ep_size != 1) {
            throw std::invalid_argument("Pipeline parallel MVP does not support expert parallelism");
        }

        const std::string model_type = this->model_config_->get<std::string>("model_type");
        static const std::unordered_set<std::string> supported_pp_models = {
            "llama", "qwen2", "qwen3"};
        if (supported_pp_models.count(model_type) == 0) {
            throw std::invalid_argument("Pipeline parallel MVP only supports llama, qwen2, and qwen3 model types");
        }
        const size_t num_hidden_layers = this->model_config_->get<size_t>("num_hidden_layers");
        if (num_hidden_layers < pp_size) {
            throw std::invalid_argument("Pipeline parallel size must not exceed the number of model layers");
        }

        pipeline_transport_ = std::make_unique<PeerCopyTransport>();
        spdlog::info(
            "Pipeline activation transport: gpu-peer-copy (synchronous MVP)");
    }

    const size_t tp_size = dist_config.tp_device_ids.size();
    const auto &model_json = this->model_config_->get_config_json();
    if (tp_size > 1 && model_json.contains("num_attention_heads")) {
        const size_t num_attention_heads =
            this->model_config_->get<size_t>("num_attention_heads");
        if (num_attention_heads % tp_size != 0) {
            throw std::invalid_argument(
                "num_attention_heads (" + std::to_string(num_attention_heads)
                + ") must be divisible by tensor parallel size ("
                + std::to_string(tp_size) + ")");
        }
    }
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
    const bool pipeline_parallel =
        communication_group_.get_dist_config().pp_device_ids.size() > 1;
    if (pipeline_parallel && strict) {
        const auto keys = state_dict_keys();
        const std::unordered_set<std::string> known_keys(keys.begin(), keys.end());
        for (const auto &[name, _] : params) {
            if (known_keys.count(name) == 0) {
                throw std::runtime_error(
                    "Parameter '" + name + "' not found in pipeline model.");
            }
        }
    }

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
        to_device(target_hidden_states),
        to_device(pp_hidden_states),
        to_device(pp_residual)};

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
    const size_t pp_size = communication_group_.get_dist_config().pp_device_ids.size();
    if (pp_size == 1) {
        // Preserve the existing TP path: every rank executes the same input concurrently.
        for (auto &worker : workers_) {
            worker->run(input);
        }
        for (auto &worker : workers_) {
            worker->wait();
        }
        return workers_[0]->get_output();
    }

    Input stage_input = input;
    Output stage_output;
    for (size_t stage = 0; stage < workers_.size(); ++stage) {
        auto &worker = workers_[stage];
        worker->run(stage_input);
        worker->wait();
        stage_output = worker->get_output();

        if (stage + 1 < workers_.size()) {
            if (!stage_output.hidden_states || !stage_output.residual) {
                throw std::runtime_error("Pipeline stage did not return hidden states and residual");
            }

            if (pipeline_transport_ == nullptr) {
                throw std::logic_error("Pipeline transport is not initialized");
            }
            const auto source_device = communication_group_
                                           .get_rank_info(static_cast<int>(stage))
                                           .device;
            if (stage_output.hidden_states->device() != source_device
                || stage_output.residual->device() != source_device) {
                throw std::runtime_error(
                    "Pipeline stage returned activations on the wrong device");
            }
            auto activation = pipeline_transport_->transfer(
                {stage_output.hidden_states, stage_output.residual},
                communication_group_.get_rank_info(static_cast<int>(stage + 1)).device);

            // Keep only metadata needed by all stages and drop first-stage-only inputs.
            stage_input.input_ids.reset();
            stage_input.pixel_values.reset();
            stage_input.image_bound.reset();
            stage_input.tgt_sizes.reset();
            stage_input.image_grid_thw.reset();
            stage_input.image_req_ids.reset();
            stage_input.visual_token_ranges.reset();
            stage_input.target_hidden_states.reset();
            stage_input.pp_hidden_states = std::move(activation.hidden_states);
            stage_input.pp_residual = std::move(activation.residual);
        }
    }

    return stage_output;
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

PipelineTransportStats InferEngine::get_pipeline_transport_stats() const {
    if (pipeline_transport_ == nullptr) {
        return {};
    }
    return pipeline_transport_->stats();
}

//------------------------------------------------------
// reset_cache (overloaded with CacheConfig)
//------------------------------------------------------
void InferEngine::reset_cache(const cache::CacheConfig *new_config) {
    const bool pipeline_parallel =
        communication_group_.get_dist_config().pp_device_ids.size() > 1;
    if (pipeline_parallel
        && dynamic_cast<const cache::PagedKVCacheConfig *>(new_config) == nullptr) {
        throw std::invalid_argument(
            "Pipeline parallel MVP requires paged KV cache");
    }

    for (auto &worker : workers_) {
        worker->reset_cache(new_config);
    }
    for (auto &worker : workers_) {
        worker->wait();
    }
    cache_config_ = new_config->unique_copy();
    this->compile();
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
