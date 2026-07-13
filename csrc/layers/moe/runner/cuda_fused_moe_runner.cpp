#include "cuda_fused_moe_runner.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/moe_align.hpp"
#include "infinicore/ops/moe_fused_dense.hpp"
#include "infinicore/ops/moe_w16a16_marlin.hpp"
#include "infinicore/ops/moe_w8a8_marlin.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace infinilm::layers::moe {

struct HygonMarlinGemmConfig {
    int mode = 103;
    int delta = 1;
    size_t block_size_m = 16;
    bool found = false;
};

struct HygonW16A16MarlinRuntimeConfig {
    HygonMarlinGemmConfig gemm1;
    HygonMarlinGemmConfig gemm2;
    bool supported = false;
};

struct HygonW8A8MarlinRuntimeConfig {
    HygonMarlinGemmConfig gemm1;
    HygonMarlinGemmConfig gemm2;
    bool supported = false;
};

CudaFusedMoeRunner::CudaFusedMoeRunner(size_t num_local_experts,
                                       size_t hidden_size,
                                       size_t intermediate_size_per_partition,
                                       size_t align_block_size)
    : num_local_experts_(num_local_experts),
      hidden_size_(hidden_size),
      intermediate_size_per_partition_(intermediate_size_per_partition),
      align_block_size_(align_block_size) {}

namespace {

std::string env_or_default(const char *name, const char *default_value) {
    const char *value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? std::string(value) : std::string(default_value);
}

std::string normalize_hygon_gpu_target(std::string target, bool uppercase) {
    const auto feature_pos = target.find(':');
    if (feature_pos != std::string::npos) {
        target.resize(feature_pos);
    }
    std::transform(target.begin(), target.end(), target.begin(), [uppercase](unsigned char ch) {
        return static_cast<char>(uppercase ? std::toupper(ch) : std::tolower(ch));
    });

    std::string lowercase = target;
    std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (lowercase.size() <= 3 || lowercase.compare(0, 3, "gfx") != 0 ||
        !std::all_of(lowercase.begin() + 3, lowercase.end(), [](unsigned char ch) {
            return std::isalnum(ch) != 0;
        })) {
        throw std::runtime_error("Invalid Hygon GPU target for lightop config: " + target);
    }
    return target;
}

constexpr size_t kHygonW8A8MoeSliceTokens = 16384;

enum class HygonMarlinModePolicy {
    LegacyOnly,
    LegacyAndBf16Mode1000,
    All,
};

HygonMarlinGemmConfig load_lightop_marlin_config(size_t n,
                                                 size_t k,
                                                 size_t m,
                                                 const std::string &file_prefix,
                                                 const infinicore::adaptor::lightop::DeviceInfo &device_info,
                                                 HygonMarlinModePolicy mode_policy,
                                                 bool uppercase_device_name,
                                                 bool num_cus_with_cu_prefix) {
    HygonMarlinGemmConfig result;
    const std::string config_dir = env_or_default(
        "INFINILM_LIGHTOP_CONFIG_DIR",
        "/usr/local/lib/python3.10/dist-packages/lightop/configs");
    if (device_info.gpu_target.empty() || device_info.compute_units <= 0) {
        throw std::runtime_error("Unable to query Hygon device properties for lightop config");
    }
    const std::string device_name = normalize_hygon_gpu_target(
        device_info.gpu_target,
        uppercase_device_name);
    const std::string num_cus = std::to_string(device_info.compute_units);
    const std::string num_cus_suffix = num_cus_with_cu_prefix ? ("_CU" + num_cus) : ("_" + num_cus);
    const std::string file_name = config_dir + "/" + file_prefix + "_" +
                                  std::to_string(n) + "_" + std::to_string(k) + "_" +
                                  device_name + num_cus_suffix + ".json";
    std::ifstream file(file_name);
    if (!file.is_open()) {
        return result;
    }

    nlohmann::json config_json;
    file >> config_json;
    const std::string shape_key = std::to_string(n) + "_" + std::to_string(k);
    if (!config_json.contains(shape_key) || !config_json.at(shape_key).is_object()) {
        return result;
    }
    const auto &configs = config_json.at(shape_key);

    auto usable = [&](size_t token) -> bool {
        const auto key = std::to_string(token);
        if (!configs.contains(key) || !configs.at(key).is_object()) {
            return false;
        }
        const int mode = configs.at(key).value("MODE", result.mode);
        return mode < 1000 ||
               mode_policy == HygonMarlinModePolicy::All ||
               (mode_policy == HygonMarlinModePolicy::LegacyAndBf16Mode1000 && mode == 1000);
    };

    size_t chosen = 0;
    bool has_choice = false;
    size_t chosen_ge = std::numeric_limits<size_t>::max();
    size_t closest_diff = std::numeric_limits<size_t>::max();
    for (auto it = configs.begin(); it != configs.end(); ++it) {
        size_t token = 0;
        try {
            token = static_cast<size_t>(std::stoull(it.key()));
        } catch (const std::exception &) {
            continue;
        }
        if (!usable(token)) {
            continue;
        }
        if (token >= m && token < chosen_ge) {
            chosen_ge = token;
            chosen = token;
            has_choice = true;
        }
        const size_t diff = token > m ? token - m : m - token;
        if (diff < closest_diff) {
            closest_diff = diff;
            if (chosen_ge == std::numeric_limits<size_t>::max()) {
                chosen = token;
                has_choice = true;
            }
        }
    }
    if (!has_choice) {
        return result;
    }

    const auto &cfg = configs.at(std::to_string(chosen));
    result.mode = cfg.value("MODE", result.mode);
    result.delta = cfg.value("DELTA", result.delta);
    result.block_size_m = cfg.value("BLOCK_SIZE_M", result.block_size_m);
    result.found = cfg.contains("MODE");
    return result;
}
HygonW16A16MarlinRuntimeConfig select_hygon_w16a16_marlin_config(size_t m,
                                                                 size_t hidden_size,
                                                                 size_t intermediate_size_per_partition,
                                                                 infinicore::DataType hidden_dtype,
                                                                 size_t device_index) {
    HygonW16A16MarlinRuntimeConfig config;
    const auto device_info = infinicore::adaptor::lightop::device_info(device_index);
    const auto mode_policy = hidden_dtype == infinicore::DataType::BF16
                                 ? HygonMarlinModePolicy::LegacyAndBf16Mode1000
                                 : HygonMarlinModePolicy::LegacyOnly;
    config.gemm1 = load_lightop_marlin_config(
        intermediate_size_per_partition * 2, hidden_size, m,
        "MOE_W16A16_CUDA_MARLIN", device_info, mode_policy, false, false);
    config.gemm2 = load_lightop_marlin_config(
        hidden_size, intermediate_size_per_partition, m,
        "MOE_W16A16_CUDA_MARLIN", device_info, mode_policy, false, false);
    config.supported = config.gemm1.found && config.gemm2.found;
    return config;
}

HygonW8A8MarlinRuntimeConfig select_hygon_w8a8_marlin_config(size_t m,
                                                             size_t hidden_size,
                                                             size_t intermediate_size_per_partition,
                                                             size_t device_index) {
    HygonW8A8MarlinRuntimeConfig config;
    const auto device_info = infinicore::adaptor::lightop::device_info(device_index);
    config.gemm1 = load_lightop_marlin_config(
        intermediate_size_per_partition * 2, hidden_size, m,
        "MOE_BLOCKINT8_CUDA_MARLIN", device_info, HygonMarlinModePolicy::All, true, true);
    config.gemm2 = load_lightop_marlin_config(
        hidden_size, intermediate_size_per_partition, m,
        "MOE_BLOCKINT8_CUDA_MARLIN", device_info, HygonMarlinModePolicy::All, true, true);
    config.supported = config.gemm1.found && config.gemm2.found;
    return config;
}

bool same_device(const infinicore::Tensor &tensor, const infinicore::Device &device) {
    return tensor && tensor->device().getType() == device.getType() && tensor->device().getIndex() == device.getIndex();
}

void ensure_tensor(infinicore::Tensor &tensor,
                   const infinicore::Shape &shape,
                   infinicore::DataType dtype,
                   const infinicore::Device &device) {
    if (!same_device(tensor, device) || tensor->dtype() != dtype || tensor->shape() != shape) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE runner workspace tensor was not initialized before graph capture");
        }
        tensor = infinicore::Tensor::empty(shape, dtype, device);
    }
}

std::string shape_to_string(const infinicore::Shape &shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

void check_packed_weight_tensor(const infinicore::Tensor &tensor,
                                const std::string &name,
                                const infinicore::Device &device,
                                const infinicore::DataType dtype,
                                const infinicore::Shape &shape) {
    if (!tensor) {
        throw std::runtime_error("MoE fused dense core requires " + name);
    }
    if (tensor->device().getType() != device.getType() || tensor->device().getIndex() != device.getIndex()) {
        throw std::runtime_error("MoE fused dense core requires packed weights on the hidden_states device");
    }
    if (tensor->dtype() != dtype) {
        throw std::runtime_error("MoE fused dense core packed tensor dtype mismatch for " + name);
    }
    if (tensor->shape() != shape) {
        throw std::runtime_error(
            "MoE fused dense core packed weight shape mismatch for " + name + ": expected " + shape_to_string(shape) + ", got " + shape_to_string(tensor->shape()));
    }
}

} // namespace

CombineInput CudaFusedMoeRunner::run(const DispatchOutput &dispatch_output,
                                     const MoeWeights &weights,
                                     MoeWorkspace &workspace) const {
    size_t block_size = align_block_size_;
    HygonW16A16MarlinRuntimeConfig marlin_config;
    HygonW8A8MarlinRuntimeConfig w8a8_marlin_config;
    if (weights.is_hygon_w16a16_marlin() || weights.is_hygon_w8a8_marlin()) {
        const auto &hidden_shape = dispatch_output.hidden_states->shape();
        if (hidden_shape.size() != 2) {
            throw std::runtime_error("Hygon Marlin MoE runner requires hidden states [M, K]");
        }
        if (weights.is_hygon_w16a16_marlin()) {
            marlin_config = select_hygon_w16a16_marlin_config(
                hidden_shape[0], hidden_size_, intermediate_size_per_partition_,
                dispatch_output.hidden_states->dtype(),
                dispatch_output.hidden_states->device().getIndex());
            if (!marlin_config.supported) {
                throw std::runtime_error("No lightop W16A16 Marlin MoE config found for this Hygon shape");
            }
            block_size = marlin_config.gemm1.block_size_m;
        } else {
            if (hidden_shape[0] > kHygonW8A8MoeSliceTokens) {
                auto runner_output = run_hygon_w8a8_marlin_core_sliced(
                    dispatch_output, weights, workspace);
                return CombineInput{
                    CombineInputFormat::Standard,
                    runner_output.hidden_states,
                    dispatch_output.topk_output,
                    MoeRoutingMetadata{},
                };
            }
            w8a8_marlin_config = select_hygon_w8a8_marlin_config(
                hidden_shape[0], hidden_size_, intermediate_size_per_partition_,
                dispatch_output.hidden_states->device().getIndex());
            if (!w8a8_marlin_config.supported) {
                throw std::runtime_error("No lightop W8A8 Marlin MoE config found for this Hygon shape");
            }
            block_size = w8a8_marlin_config.gemm1.block_size_m;
        }
    }

    auto runner_input = prepare_runner_input(
        dispatch_output,
        workspace,
        block_size);

    auto runner_output = weights.is_hygon_w16a16_marlin()
                             ? run_hygon_w16a16_marlin_core(runner_input, weights, workspace, marlin_config)
                             : (weights.is_hygon_w8a8_marlin()
                                    ? run_hygon_w8a8_marlin_core(
                                          runner_input, weights, workspace, w8a8_marlin_config)
                                    : run_fused_core(runner_input, weights, workspace));

    return CombineInput{
        CombineInputFormat::Standard,
        runner_output.hidden_states,
        runner_input.topk_output,
        runner_input.routing_metadata,
    };
}

CudaFusedMoeRunnerInput CudaFusedMoeRunner::prepare_runner_input(const DispatchOutput &dispatch_output,
                                                                 MoeWorkspace &workspace,
                                                                 size_t block_size) const {
    const auto &topk_ids = dispatch_output.topk_output.topk_ids;
    const auto &topk_shape = topk_ids->shape();
    if (topk_shape.size() != 2) {
        throw std::runtime_error("MoE runner requires topk_ids to be a 2D tensor");
    }
    const size_t num_pairs = topk_shape[0] * topk_shape[1];
    const size_t align_num_experts = num_local_experts_ + 1;
    const size_t max_num_tokens_padded = num_pairs < align_num_experts
                                           ? num_pairs * block_size
                                           : num_pairs + align_num_experts * (block_size - 1);
    const size_t sorted_token_ids_capacity = ((max_num_tokens_padded + 3) / 4) * 4;
    const size_t max_num_blocks = (max_num_tokens_padded + block_size - 1) / block_size;
    const auto device = topk_ids->device();

    if (!workspace.sorted_token_ids || workspace.sorted_token_ids_capacity < sorted_token_ids_capacity) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE sorted_token_ids workspace was not initialized before graph capture");
        }
        workspace.sorted_token_ids = infinicore::Tensor::empty(
            {sorted_token_ids_capacity}, infinicore::DataType::I32, device);
        workspace.sorted_token_ids_capacity = sorted_token_ids_capacity;
    }
    if (!workspace.expert_ids || workspace.expert_ids_capacity < max_num_blocks) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE expert_ids workspace was not initialized before graph capture");
        }
        workspace.expert_ids = infinicore::Tensor::empty(
            {max_num_blocks}, infinicore::DataType::I32, device);
        workspace.expert_ids_capacity = max_num_blocks;
    }
    if (!workspace.num_tokens_post_padded) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE num_tokens_post_padded workspace was not initialized before graph capture");
        }
        workspace.num_tokens_post_padded = infinicore::Tensor::empty(
            {1}, infinicore::DataType::I32, device);
    }

    if (dispatch_output.expert_map) {
        infinicore::op::moe_align_with_expert_map_(
            workspace.sorted_token_ids,
            workspace.expert_ids,
            workspace.num_tokens_post_padded,
            topk_ids,
            dispatch_output.expert_map,
            num_local_experts_,
            block_size,
            true);
    } else {
        infinicore::op::moe_align_(
            workspace.sorted_token_ids,
            workspace.expert_ids,
            workspace.num_tokens_post_padded,
            topk_ids,
            num_local_experts_,
            block_size,
            true);
    }
    return CudaFusedMoeRunnerInput{
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
        MoeRoutingMetadata{
            workspace.sorted_token_ids,
            workspace.expert_ids,
            workspace.num_tokens_post_padded,
        },
    };
}

CudaFusedMoeRunnerOutput CudaFusedMoeRunner::run_fused_core(const CudaFusedMoeRunnerInput &runner_input,
                                                            const MoeWeights &weights,
                                                            MoeWorkspace &workspace) const {
    if (!weights.has_packed_dense_weights()) {
        throw std::runtime_error("MoE fused dense runner requires load-time packed w13/w2 weights");
    }
    check_packed_weight_tensor(
        weights.packed_w13,
        "w13",
        runner_input.hidden_states->device(),
        runner_input.hidden_states->dtype(),
        {num_local_experts_, intermediate_size_per_partition_ * 2, hidden_size_});
    check_packed_weight_tensor(
        weights.packed_w2,
        "w2",
        runner_input.hidden_states->device(),
        runner_input.hidden_states->dtype(),
        {num_local_experts_, hidden_size_, intermediate_size_per_partition_});
    ensure_tensor(
        workspace.fused_moe_output,
        runner_input.hidden_states->shape(),
        runner_input.hidden_states->dtype(),
        runner_input.hidden_states->device());
    infinicore::op::moe_fused_dense_(
        workspace.fused_moe_output,
        runner_input.hidden_states,
        weights.packed_w13,
        weights.packed_w2,
        runner_input.topk_output.topk_weights,
        runner_input.topk_output.topk_ids,
        runner_input.routing_metadata.sorted_token_ids,
        runner_input.routing_metadata.expert_ids,
        runner_input.routing_metadata.num_tokens_post_padded);
    return CudaFusedMoeRunnerOutput{
        workspace.fused_moe_output,
    };
}

CudaFusedMoeRunnerOutput CudaFusedMoeRunner::run_hygon_w16a16_marlin_core(
    const CudaFusedMoeRunnerInput &runner_input,
    const MoeWeights &weights,
    MoeWorkspace &workspace,
    const HygonW16A16MarlinRuntimeConfig &config) const {
    if (!weights.has_packed_dense_weights() || !weights.is_hygon_w16a16_marlin()) {
        throw std::runtime_error("Hygon W16A16 Marlin MoE runner requires packed Marlin weights");
    }
    const auto activation_dtype = runner_input.hidden_states->dtype();
    if (activation_dtype != infinicore::DataType::BF16 &&
        activation_dtype != infinicore::DataType::F16) {
        throw std::runtime_error("Hygon W16A16 Marlin MoE runner requires BF16 or FP16 activations");
    }
    check_packed_weight_tensor(
        weights.packed_w13,
        "w13",
        runner_input.hidden_states->device(),
        activation_dtype,
        {num_local_experts_, hidden_size_ / 16, intermediate_size_per_partition_ * 2 * 16});
    check_packed_weight_tensor(
        weights.packed_w2,
        "w2",
        runner_input.hidden_states->device(),
        activation_dtype,
        {num_local_experts_, intermediate_size_per_partition_ / 16, hidden_size_ * 16});
    const size_t top_k = runner_input.topk_output.topk_ids->shape()[1];
    const size_t num_tokens = runner_input.hidden_states->shape()[0];
    const size_t cache13_required = num_tokens * top_k * std::max(intermediate_size_per_partition_ * 2, hidden_size_);
    const size_t cache2_required = num_tokens * top_k * intermediate_size_per_partition_;

    ensure_tensor(
        workspace.fused_moe_output,
        runner_input.hidden_states->shape(),
        runner_input.hidden_states->dtype(),
        runner_input.hidden_states->device());
    if (!same_device(workspace.marlin_cache13, runner_input.hidden_states->device()) ||
        workspace.marlin_cache13->dtype() != runner_input.hidden_states->dtype() ||
        workspace.marlin_cache13_capacity < cache13_required) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE Marlin cache13 workspace was not initialized before graph capture");
        }
        workspace.marlin_cache13 = infinicore::Tensor::empty(
            {cache13_required}, runner_input.hidden_states->dtype(), runner_input.hidden_states->device());
        workspace.marlin_cache13_capacity = cache13_required;
    }
    if (!same_device(workspace.marlin_cache2, runner_input.hidden_states->device()) ||
        workspace.marlin_cache2->dtype() != runner_input.hidden_states->dtype() ||
        workspace.marlin_cache2_capacity < cache2_required) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE Marlin cache2 workspace was not initialized before graph capture");
        }
        workspace.marlin_cache2 = infinicore::Tensor::empty(
            {cache2_required}, runner_input.hidden_states->dtype(), runner_input.hidden_states->device());
        workspace.marlin_cache2_capacity = cache2_required;
    }

    infinicore::op::moe_w16a16_marlin_fused_dense_(
        workspace.fused_moe_output,
        workspace.marlin_cache13,
        workspace.marlin_cache2,
        runner_input.hidden_states,
        weights.packed_w13,
        weights.packed_w2,
        runner_input.topk_output.topk_weights,
        runner_input.routing_metadata.sorted_token_ids,
        runner_input.routing_metadata.expert_ids,
        runner_input.routing_metadata.num_tokens_post_padded,
        top_k,
        config.gemm1.mode,
        config.gemm1.delta,
        config.gemm2.mode,
        config.gemm2.delta);

    return CudaFusedMoeRunnerOutput{
        workspace.fused_moe_output,
    };
}

CudaFusedMoeRunnerOutput CudaFusedMoeRunner::run_hygon_w8a8_marlin_core(
    const CudaFusedMoeRunnerInput &runner_input,
    const MoeWeights &weights,
    MoeWorkspace &workspace,
    const HygonW8A8MarlinRuntimeConfig &config) const {
    if (!weights.has_packed_w8a8_marlin_weights() || !weights.is_hygon_w8a8_marlin()) {
        throw std::runtime_error("Hygon W8A8 Marlin MoE runner requires packed Marlin weights and scales");
    }
    const size_t top_k = runner_input.topk_output.topk_ids->shape()[1];
    const size_t num_tokens = runner_input.hidden_states->shape()[0];
    const size_t cache13_required = num_tokens * top_k * std::max(intermediate_size_per_partition_ * 2, hidden_size_);

    check_packed_weight_tensor(
        weights.packed_w13,
        "w13",
        runner_input.hidden_states->device(),
        infinicore::DataType::I8,
        {num_local_experts_, hidden_size_ / 64, intermediate_size_per_partition_ * 2 * 64});
    check_packed_weight_tensor(
        weights.packed_w2,
        "w2",
        runner_input.hidden_states->device(),
        infinicore::DataType::I8,
        {num_local_experts_, intermediate_size_per_partition_ / 64, hidden_size_ * 64});
    check_packed_weight_tensor(
        weights.packed_w13_scale,
        "w13_scale",
        runner_input.hidden_states->device(),
        infinicore::DataType::F32,
        {num_local_experts_, intermediate_size_per_partition_ * 2, 1});
    check_packed_weight_tensor(
        weights.packed_w2_scale,
        "w2_scale",
        runner_input.hidden_states->device(),
        infinicore::DataType::F32,
        {num_local_experts_, hidden_size_, 1});

    ensure_tensor(
        workspace.fused_moe_output,
        runner_input.hidden_states->shape(),
        runner_input.hidden_states->dtype(),
        runner_input.hidden_states->device());
    if (!same_device(workspace.marlin_cache13, runner_input.hidden_states->device()) ||
        workspace.marlin_cache13->dtype() != runner_input.hidden_states->dtype() ||
        workspace.marlin_cache13_capacity < cache13_required) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE W8A8 Marlin cache13 workspace was not initialized before graph capture");
        }
        workspace.marlin_cache13 = infinicore::Tensor::empty(
            {cache13_required}, runner_input.hidden_states->dtype(), runner_input.hidden_states->device());
        workspace.marlin_cache13_capacity = cache13_required;
    }
    ensure_tensor(
        workspace.marlin_input_i8,
        {num_tokens, hidden_size_},
        infinicore::DataType::I8,
        runner_input.hidden_states->device());
    ensure_tensor(
        workspace.marlin_input_scale,
        {num_tokens, 1},
        infinicore::DataType::F32,
        runner_input.hidden_states->device());
    ensure_tensor(
        workspace.marlin_cache2_i8,
        {num_tokens * top_k, intermediate_size_per_partition_},
        infinicore::DataType::I8,
        runner_input.hidden_states->device());
    ensure_tensor(
        workspace.marlin_cache2_scale,
        {num_tokens * top_k, 1},
        infinicore::DataType::F32,
        runner_input.hidden_states->device());

    infinicore::op::moe_w8a8_marlin_fused_dense_(
        workspace.fused_moe_output,
        workspace.marlin_cache13,
        workspace.marlin_cache2_i8,
        workspace.marlin_input_i8,
        workspace.marlin_input_scale,
        workspace.marlin_cache2_scale,
        runner_input.hidden_states,
        weights.packed_w13,
        weights.packed_w2,
        weights.packed_w13_scale,
        weights.packed_w2_scale,
        runner_input.topk_output.topk_weights,
        runner_input.routing_metadata.sorted_token_ids,
        runner_input.routing_metadata.expert_ids,
        runner_input.routing_metadata.num_tokens_post_padded,
        top_k,
        config.gemm1.mode,
        config.gemm1.block_size_m,
        config.gemm1.delta,
        config.gemm2.mode,
        config.gemm2.delta);

    return CudaFusedMoeRunnerOutput{
        workspace.fused_moe_output,
    };
}

CudaFusedMoeRunnerOutput CudaFusedMoeRunner::run_hygon_w8a8_marlin_core_sliced(
    const DispatchOutput &dispatch_output,
    const MoeWeights &weights,
    MoeWorkspace &workspace) const {
    const auto &hidden_shape = dispatch_output.hidden_states->shape();
    if (hidden_shape.size() != 2) {
        throw std::runtime_error("Hygon W8A8 sliced MoE runner requires hidden states [M, K]");
    }
    const size_t num_tokens = hidden_shape[0];
    if (infinicore::context::isGraphRecording()) {
        throw std::runtime_error("Hygon W8A8 sliced MoE runner cannot allocate/copy slice outputs during graph capture");
    }

    ensure_tensor(
        workspace.fused_moe_output,
        dispatch_output.hidden_states->shape(),
        dispatch_output.hidden_states->dtype(),
        dispatch_output.hidden_states->device());
    MoeWorkspace slice_workspace;
    const auto device_index = dispatch_output.hidden_states->device().getIndex();
    const auto full_slice_config = select_hygon_w8a8_marlin_config(
        kHygonW8A8MoeSliceTokens, hidden_size_, intermediate_size_per_partition_,
        device_index);
    if (!full_slice_config.supported) {
        throw std::runtime_error("No lightop W8A8 Marlin MoE config found for full Hygon slice");
    }
    size_t offset = 0;
    while (offset < num_tokens) {
        const size_t slice_tokens = std::min(kHygonW8A8MoeSliceTokens, num_tokens - offset);
        const auto slice_config = slice_tokens == kHygonW8A8MoeSliceTokens
                                    ? full_slice_config
                                    : select_hygon_w8a8_marlin_config(
                                          slice_tokens, hidden_size_, intermediate_size_per_partition_,
                                          device_index);
        if (!slice_config.supported) {
            throw std::runtime_error("No lightop W8A8 Marlin MoE config found for sliced Hygon shape");
        }

        const auto hidden_slice = dispatch_output.hidden_states->narrow({{0, offset, slice_tokens}});
        const TopKOutput topk_slice{
            dispatch_output.topk_output.topk_weights->narrow({{0, offset, slice_tokens}}),
            dispatch_output.topk_output.topk_ids->narrow({{0, offset, slice_tokens}}),
            dispatch_output.topk_output.router_logits
                ? dispatch_output.topk_output.router_logits->narrow({{0, offset, slice_tokens}})
                : infinicore::Tensor(),
        };
        const DispatchOutput dispatch_slice{
            DispatchOutputFormat::Standard,
            hidden_slice,
            infinicore::Tensor(),
            topk_slice,
            infinicore::Tensor(),
        };
        auto slice_input = prepare_runner_input(
            dispatch_slice,
            slice_workspace,
            slice_config.gemm1.block_size_m);
        auto slice_output = run_hygon_w8a8_marlin_core(
            slice_input,
            weights,
            slice_workspace,
            slice_config);
        workspace.fused_moe_output->narrow({{0, offset, slice_tokens}})->copy_from(slice_output.hidden_states);

        offset += slice_tokens;
    }

    return CudaFusedMoeRunnerOutput{
        workspace.fused_moe_output,
    };
}

} // namespace infinilm::layers::moe
