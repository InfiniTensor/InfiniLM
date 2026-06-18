#include "ep_config.hpp"

#include "../../../global_state/parallel_state.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace infinilm::layers::moe {

namespace {

std::string lower_string(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

EPBackend parse_backend(const std::string &backend) {
    if (backend.empty() || backend == "0" || backend == "none" || backend == "off" || backend == "disabled" || backend == "standard") {
        return EPBackend::Disabled;
    }
    if (backend == "naive" || backend == "ag_rs" || backend == "allgather_reducescatter" || backend == "all_gather_reduce_scatter") {
        return EPBackend::AllGatherReduceScatter;
    }
    if (backend == "local_allreduce" || backend == "local_all_reduce" || backend == "tp_ep" || backend == "vllm_tp" || backend == "dp1") {
        return EPBackend::LocalAllReduce;
    }
    if (backend == "deepep" || backend == "deep_ep") {
        return EPBackend::DeepEP;
    }
    throw std::runtime_error("Unsupported MoE EP backend: " + backend);
}

DeepEPMode parse_deepep_mode(const std::string &mode) {
    if (mode.empty() || mode == "auto") {
        return DeepEPMode::Auto;
    }
    if (mode == "normal") {
        return DeepEPMode::Normal;
    }
    if (mode == "low_latency" || mode == "lowlatency" || mode == "ll") {
        return DeepEPMode::LowLatency;
    }
    throw std::runtime_error("Unsupported MoE DeepEP mode: " + mode);
}

} // namespace

const char *ep_backend_name(EPBackend backend) {
    switch (backend) {
    case EPBackend::Disabled:
        return "disabled";
    case EPBackend::AllGatherReduceScatter:
        return "allgather_reducescatter";
    case EPBackend::LocalAllReduce:
        return "local_allreduce";
    case EPBackend::DeepEP:
        return "deepep";
    }
    return "unknown";
}

EPConfig make_ep_config(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    EPConfig config;
    auto &config_json = model_config->get_config_json();
    const std::string backend = config_json.contains("moe_ep_backend") && !config_json["moe_ep_backend"].is_null()
                                  ? lower_string(config_json["moe_ep_backend"].get<std::string>())
                                  : std::string();
    config.backend = parse_backend(backend);

    if (config.backend == EPBackend::Disabled) {
        return config;
    }

    const std::string deepep_mode = config_json.contains("moe_deepep_mode") && !config_json["moe_deepep_mode"].is_null()
                                      ? lower_string(config_json["moe_deepep_mode"].get<std::string>())
                                      : std::string();
    config.deepep_mode = parse_deepep_mode(deepep_mode);

    const size_t tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    const size_t tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    config.ep_size = config_json.contains("moe_ep_size") && !config_json["moe_ep_size"].is_null()
                       ? config_json["moe_ep_size"].get<size_t>()
                       : tp_size;
    config.ep_rank = tp_rank;

    if (config.ep_size == 0) {
        throw std::runtime_error("MoE EP size must be greater than 0");
    }
    if (config.ep_rank >= config.ep_size) {
        throw std::runtime_error("MoE EP rank must be smaller than EP size");
    }
    return config;
}

ExpertPlacement make_expert_placement(const EPConfig &config,
                                      size_t global_num_experts) {
    if (global_num_experts == 0) {
        throw std::runtime_error("MoE expert placement requires at least one expert");
    }
    if (config.backend == EPBackend::Disabled || config.ep_size == 1) {
        return ExpertPlacement{
            global_num_experts,
            global_num_experts,
            0,
            global_num_experts,
        };
    }
    if (config.ep_size == 0) {
        throw std::runtime_error("MoE EP size must be greater than 0");
    }
    if (config.ep_rank >= config.ep_size) {
        throw std::runtime_error("MoE EP rank must be smaller than EP size");
    }

    const size_t base_experts = global_num_experts / config.ep_size;
    const size_t remainder = global_num_experts % config.ep_size;
    const size_t local_num_experts = base_experts + (config.ep_rank < remainder ? 1 : 0);
    const size_t local_expert_start = config.ep_rank * base_experts + std::min(config.ep_rank, remainder);
    return ExpertPlacement{
        global_num_experts,
        local_num_experts,
        local_expert_start,
        local_expert_start + local_num_experts,
    };
}

} // namespace infinilm::layers::moe
