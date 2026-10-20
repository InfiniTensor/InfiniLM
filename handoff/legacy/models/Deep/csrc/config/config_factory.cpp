#include "config_factory.hpp"
#include "../models/models_registry.hpp"
#include <cstdint>
#include <stdexcept>
#include <unordered_set>

namespace infinilm::config {

std::shared_ptr<infinilm::config::ModelConfig> ConfigFactory::createConfig(const std::string &config_str) {
    const nlohmann::json config_json = nlohmann::json::parse(config_str);
    auto model_config = std::make_shared<infinilm::config::ModelConfig>(config_json);

    const std::string model_type = model_config->get<std::string>("model_type");
    const auto quant_scheme = model_config->get_quant_scheme();
    if (model_type == "qwen3_moe"
        && (model_config->get_dtype() != infinilm::runtime::DataType::kFloat16
            || quant_scheme != quantization::QuantScheme::NONE
            || model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
            || (config_json.contains("torch_dtype")
                && parse_dtype(config_json.at("torch_dtype").get<std::string>()) != infinilm::runtime::DataType::kFloat16))) {
        throw std::invalid_argument("Public Qwen3MoE requires FP16 and NONE weight/KV quantization");
    }
    if (quant_scheme == quantization::QuantScheme::AWQ_W4A16
        || quant_scheme == quantization::QuantScheme::GPTQ_W4A16) {
        if (model_type != "qwen3" || model_config->get_dtype() != infinilm::runtime::DataType::kFloat16
            || model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
            || (config_json.contains("torch_dtype")
                && parse_dtype(config_json.at("torch_dtype").get<std::string>()) != infinilm::runtime::DataType::kFloat16))
            throw std::invalid_argument(std::string("Public ")
                + (quant_scheme == quantization::QuantScheme::AWQ_W4A16 ? "AWQ" : "GPTQ")
                + " requires exact Qwen3, FP16 and NONE KV quantization");
    }
    if (model_type == "minicpm_sala") {
        throw std::invalid_argument(
            "minicpm_sala is not supported: its InfLLMv2 and Lightning attention "
            "implementations are placeholders");
    }
    const auto &config_map = models::get_model_config_map();
    auto it = config_map.find(model_type);
    if (it == config_map.end()) {
        throw std::invalid_argument("infinilm::config::ConfigFactory::createConfig: Unsupported model config type: " + model_type);
    }

    static const std::unordered_set<std::string> kModernModelTypes{
        "deepseek_v4",
        "glm_moe_dsa",
        "baichuan",
        "chatglm",
        "deepseek",
        "ernie4_5_moe_vl",
        "fm9g",
        "fm9g7b",
        "glm4",
        "gpt2",
        "internlm3",
        "llama",
        "mamba",
        "mistral",
        "minicpm",
        "minicpm4",
        "minicpm_eagle",
        "qwen2",
        "qwen3",
        "qwen3_moe"};
    if (kModernModelTypes.find(model_type) == kModernModelTypes.end()) {
        throw std::invalid_argument(
            "infinilm::config::ConfigFactory::createConfig: model type `" + model_type
            + "` is unavailable with the modern InfiniOps backend; supported model types: "
              "baichuan, chatglm, deepseek, ernie4_5_moe_vl, fm9g, fm9g7b, glm4, gpt2, internlm3, llama, mamba, mistral, minicpm, minicpm4, minicpm_eagle, qwen2, qwen3, qwen3_moe");
    }

    it->second(model_config);
    if (model_type == "ernie4_5_moe_vl") {
        const auto &cfg = model_config->get_config_json();
        if (model_config->get_dtype() != infinilm::runtime::DataType::kFloat16
            || quant_scheme != quantization::QuantScheme::NONE
            || model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
            || (config_json.contains("torch_dtype")
                && parse_dtype(config_json.at("torch_dtype").get<std::string>()) != infinilm::runtime::DataType::kFloat16)
            || (cfg.contains("vision_config") && !cfg.at("vision_config").is_null())
            || (cfg.contains("text_config") && !cfg.at("text_config").is_null()))
            throw std::invalid_argument("Public Ernie requires flat text-only FP16 configuration and NONE weight/KV quantization; vision is not qualified");
        const auto groups = cfg.value("moe_num_experts", nlohmann::json());
        const auto widths = cfg.value("moe_intermediate_size", nlohmann::json());
        const auto start = cfg.value("moe_layer_start_index", nlohmann::json());
        const auto end = cfg.value("moe_layer_end_index", nlohmann::json());
        const auto pair = [](const nlohmann::json &value) {
            return value.is_array() && value.size() == 2
                && value[0].is_number_integer() && value[1].is_number_integer();
        };
        if (!pair(groups) || groups[0] != groups[1] || groups[0].get<int64_t>() <= 0
            || !pair(widths) || widths[0].get<int64_t>() <= 0 || widths[1].get<int64_t>() <= 0
            || !pair(start) || start[0] != start[1] || start[0].get<int64_t>() < 0
            || !pair(end) || end[0] != end[1] || end[0].get<int64_t>() < start[0].get<int64_t>()
            || end[0].get<int64_t>() >= cfg.at("num_hidden_layers").get<int64_t>()
            || cfg.at("num_experts") != groups[0] || cfg.value("moe_layer_interval", 1) != 1
            || cfg.value("moe_k", cfg.at("num_experts_per_tok").get<size_t>())
                != cfg.at("num_experts_per_tok").get<size_t>()
            || !cfg.value("use_moe", false) || !cfg.value("norm_topk_prob", false)
            || !cfg.value("moe_norm_gate_logits", true)
            || cfg.value("moe_capacity", 128.0) < groups[0].get<double>())
            throw std::invalid_argument("Public Ernie requires two equal expert counts, aligned MoE layer ranges, normalized routing and non-dropping capacity");
        if (cfg.value("hidden_act", std::string("silu")) != "silu"
            || cfg.value("use_bias", false) || cfg.value("mlp_bias", false)
            || cfg.value("attention_bias", false) || cfg.value("attention_output_bias", false)
            || cfg.value("hidden_dropout_prob", 0.0) != 0.0
            || cfg.value("attention_probs_dropout_prob", 0.0) != 0.0
            || cfg.value("moe_dropout_prob", 0.0) != 0.0
            || cfg.value("moe_all_to_all_dropout", 0.0) != 0.0
            || cfg.value("partial_rotary_factor", 1.0) != 1.0
            || (cfg.contains("rope_scaling") && !cfg.at("rope_scaling").is_null())
            || cfg.value("freq_allocation", 20) != 20
            || (cfg.contains("rope_parameters")
                && (!cfg.at("rope_parameters").is_object()
                    || cfg.at("rope_parameters").size() > 1
                    || (!cfg.at("rope_parameters").empty()
                        && !cfg.at("rope_parameters").contains("mrope_section"))
                    || cfg.at("rope_parameters").value("mrope_section", nlohmann::json{22, 22, 20})
                        != nlohmann::json{22, 22, 20}))
            || model_config->get_head_dim() != 128)
            throw std::invalid_argument("Public Ernie requires bias-free SiLU text inference, zero dropout and full unscaled 128-dimensional RoPE");
    }
    if (model_type == "mistral"
        && (model_config->get_dtype() != infinilm::runtime::DataType::kFloat16
            || quant_scheme != quantization::QuantScheme::NONE
            || model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
            || (config_json.contains("torch_dtype")
                && parse_dtype(config_json.at("torch_dtype").get<std::string>()) != infinilm::runtime::DataType::kFloat16)))
        throw std::invalid_argument("Public Mistral requires FP16 and NONE weight/KV quantization");
    if (model_type == "mamba"
        && (model_config->get_dtype() != infinilm::runtime::DataType::kFloat32
            || quant_scheme != quantization::QuantScheme::NONE
            || model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
            || (config_json.contains("torch_dtype")
                && parse_dtype(config_json.at("torch_dtype").get<std::string>()) != infinilm::runtime::DataType::kFloat32))) {
        throw std::invalid_argument("Public Mamba requires FP32 and NONE weight/KV quantization");
    }
    if (model_type == "mamba" && model_config->get<size_t>("conv_kernel") != 4) {
        throw std::invalid_argument("Public Mamba requires conv_kernel=4");
    }
    if (model_type == "deepseek") {
        const auto &normalized = model_config->get_config_json();
        if (model_config->get_dtype() != infinilm::runtime::DataType::kFloat16
            || quant_scheme != quantization::QuantScheme::NONE
            || model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
            || (config_json.contains("torch_dtype")
                && parse_dtype(config_json.at("torch_dtype").get<std::string>()) != infinilm::runtime::DataType::kFloat16)) {
            throw std::invalid_argument("Public Deepseek requires FP16 and NONE weight/KV quantization");
        }
        if (normalized.value("pretraining_tp", 1) != 1
            || (normalized.contains("rope_scaling") && !normalized.at("rope_scaling").is_null())
            || normalized.value("rope_theta", 10000.0) != 10000.0
            || normalized.value("attention_bias", false)
            || normalized.value("attention_output_bias", false)
            || normalized.value("attention_dropout", 0.0) != 0.0
            || normalized.value("tie_word_embeddings", false)) {
            throw std::invalid_argument("Public Deepseek requires pretraining_tp=1, unscaled RoPE with theta=10000, no attention bias/dropout and untied embeddings");
        }
        const auto routed = model_config->get<size_t>("n_routed_experts");
        if (routed == 0 && model_config->get<size_t>("n_shared_experts") != 0) {
            throw std::invalid_argument("Public Deepseek requires routed experts when shared experts are enabled");
        }
        if (routed != 0) {
            if ((normalized.contains("scoring_func")
                    && (!normalized.at("scoring_func").is_string() || normalized.at("scoring_func") != "softmax"))
                || (normalized.contains("topk_method")
                    && (!normalized.at("topk_method").is_string() || normalized.at("topk_method") != "greedy"))
                || (normalized.contains("routed_scaling_factor")
                    && (!normalized.at("routed_scaling_factor").is_number()
                        || normalized.at("routed_scaling_factor").get<double>() != 1.0))
                || (normalized.contains("n_group")
                    && (!normalized.at("n_group").is_number_integer() || normalized.at("n_group").get<size_t>() != 1))
                || (normalized.contains("topk_group")
                    && (!normalized.at("topk_group").is_number_integer() || normalized.at("topk_group").get<size_t>() != 1))) {
                throw std::invalid_argument("Public Deepseek V1 MoE requires softmax, greedy top-k, unit routed scaling and no grouped routing");
            }
            if (!normalized.at("norm_topk_prob").is_boolean()
                || (normalized.contains("first_k_dense_replace")
                    && (!normalized.at("first_k_dense_replace").is_number_integer()
                        || normalized.at("first_k_dense_replace").get<int64_t>() < 0))
                || (normalized.contains("moe_layer_freq")
                    && (!normalized.at("moe_layer_freq").is_number_integer()
                        || normalized.at("moe_layer_freq").get<int64_t>() <= 0))) {
                throw std::invalid_argument("Public Deepseek V1 MoE requires boolean norm_topk_prob, nonnegative first_k_dense_replace and positive moe_layer_freq");
            }
            if (model_config->get<size_t>("num_experts_per_tok") == 1
                && normalized.at("norm_topk_prob").get<bool>()) {
                throw std::invalid_argument("Public Deepseek V1 MoE requires norm_topk_prob=false when num_experts_per_tok=1");
            }
        }
    }
    return model_config;
}

} // namespace infinilm::config
