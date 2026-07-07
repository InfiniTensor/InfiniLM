#include "deepseek_v4_rope.hpp"

#include "infinicore/nn/rope.hpp"
#include "infinicore/nn/rope_scaling_configs.hpp"
#include "infinicore/ops/cat.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {

std::shared_ptr<infinicore::nn::RopeScalingConfig>
make_compress_yarn_scaling(const nlohmann::json &config_json,
                           size_t rotary_dim,
                           float compress_rope_theta) {
    if (!config_json.contains("rope_scaling") || !config_json.at("rope_scaling").is_object()) {
        return nullptr;
    }
    const auto &rope_scaling = config_json.at("rope_scaling");
    if (!rope_scaling.contains("factor") || !rope_scaling.contains("original_max_position_embeddings")) {
        return nullptr;
    }

    const float factor = rope_scaling.at("factor").get<float>();
    const size_t original_max_position_embeddings = rope_scaling.at("original_max_position_embeddings").get<size_t>();
    const int beta_fast = static_cast<int>(rope_scaling.value("beta_fast", 32.0f));
    const int beta_slow = static_cast<int>(rope_scaling.value("beta_slow", 1.0f));
    const float mscale = rope_scaling.value("mscale", 1.0f);
    const float mscale_all_dim = rope_scaling.value("mscale_all_dim", 0.0f);

    return std::make_shared<infinicore::nn::YarnRopeScalingConfig>(
        factor,
        original_max_position_embeddings,
        rotary_dim,
        compress_rope_theta,
        beta_fast,
        beta_slow,
        mscale,
        mscale_all_dim);
}

size_t rope_cache_max_seq_len(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                              const nlohmann::json &config_json,
                              bool use_yarn) {
    const size_t max_position_embeddings = model_config->get_or<size_t>("max_position_embeddings", 4096);
    if (!use_yarn || !config_json.contains("rope_scaling") || !config_json.at("rope_scaling").is_object()) {
        return max_position_embeddings;
    }
    const auto &rope_scaling = config_json.at("rope_scaling");
    if (!rope_scaling.contains("factor") || !rope_scaling.contains("original_max_position_embeddings")) {
        return max_position_embeddings;
    }
    const float factor = rope_scaling.at("factor").get<float>();
    const size_t original_max_position_embeddings = rope_scaling.at("original_max_position_embeddings").get<size_t>();
    return std::max(
        max_position_embeddings,
        infinicore::nn::YarnRopeScalingConfig::max_seq_len(factor, original_max_position_embeddings));
}

bool gpu_yarn_numerics_compatible(const nlohmann::json &config_json) {
    if (!config_json.contains("rope_scaling") || !config_json.at("rope_scaling").is_object()) {
        return true;
    }
    const double extrapolation_factor = config_json.at("rope_scaling").value("extrapolation_factor", 1.0);
    // CPU reference applies extrapolation_factor in deepseek_v4_yarn_inv_freq; InfiniCore YaRN has no equivalent.
    return std::fabs(extrapolation_factor - 1.0) <= 1e-7;
}

void create_gpu_ropes_if_needed(DeepseekV4RoPE::Backend backend,
                                const infinicore::Device &device,
                                const DeepseekV4RopeParams &params,
                                size_t compress_ratio,
                                std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                const nlohmann::json &config_json,
                                double rope_theta,
                                double compress_rope_theta,
                                std::shared_ptr<infinicore::nn::RoPE> &main_rope,
                                std::shared_ptr<infinicore::nn::RoPE> &compress_rope) {
    if (backend == DeepseekV4RoPE::Backend::CPU || device.getType() == infinicore::Device::Type::CPU) {
        return;
    }
    if (params.rope_dim < 2) {
        return;
    }

    const size_t head_dim = params.head_dim;
    const size_t rotary_dim = params.rope_dim;
    const auto dtype = model_config->get_dtype();
    const auto algo = model_config->get_rope_algo();

    if (compress_ratio > 1) {
        if (!gpu_yarn_numerics_compatible(config_json)) {
            return;
        }
        const size_t max_seq_len = rope_cache_max_seq_len(model_config, config_json, true);
        auto scaling = make_compress_yarn_scaling(config_json, rotary_dim, static_cast<float>(compress_rope_theta));
        compress_rope = std::make_shared<infinicore::nn::RoPE>(
            head_dim,
            rotary_dim,
            max_seq_len,
            compress_rope_theta,
            algo,
            dtype,
            device,
            scaling);
        return;
    }

    const size_t max_seq_len = rope_cache_max_seq_len(model_config, config_json, false);
    main_rope = std::make_shared<infinicore::nn::RoPE>(
        head_dim,
        rotary_dim,
        max_seq_len,
        rope_theta,
        algo,
        dtype,
        device,
        nullptr);
}

} // namespace

DeepseekV4RoPE::DeepseekV4RoPE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device,
                               Backend backend)
    : backend_(backend),
      device_(device) {
    const auto &config_json = model_config->get_config_json();
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t qk_rope_head_dim = model_config->get_or<size_t>("qk_rope_head_dim", 0);
    const double rope_theta = model_config->get_or<double>("rope_theta", 10000.0);
    const double compress_rope_theta = model_config->get_or<double>("compress_rope_theta", model_config->get_or<double>("rope_theta", 10000.0));

    size_t compress_ratio = 0;
    if (config_json.contains("compress_ratios") && layer_idx < config_json.at("compress_ratios").size()) {
        compress_ratio = config_json.at("compress_ratios").at(layer_idx).get<size_t>();
    }
    compress_ratio_ = compress_ratio;

    double yarn_factor = 1.0;
    double yarn_beta_fast = 32.0;
    double yarn_beta_slow = 1.0;
    double yarn_extrapolation_factor = 1.0;
    int64_t yarn_original_seq_len = 0;
    if (config_json.contains("rope_scaling") && config_json.at("rope_scaling").is_object()) {
        const auto &rope_scaling = config_json.at("rope_scaling");
        yarn_factor = rope_scaling.value("factor", 1.0);
        yarn_beta_fast = rope_scaling.value("beta_fast", 32.0);
        yarn_beta_slow = rope_scaling.value("beta_slow", 1.0);
        yarn_extrapolation_factor = rope_scaling.value("extrapolation_factor", 1.0);
        yarn_original_seq_len = rope_scaling.value(
            "original_max_position_embeddings",
            static_cast<int64_t>(model_config->get_or<size_t>("max_position_embeddings", 0)));
    }

    params_.head_dim = head_dim;
    params_.rope_dim = qk_rope_head_dim;
    params_.use_yarn = compress_ratio_ > 1;
    params_.rope_theta = params_.use_yarn ? compress_rope_theta : rope_theta;
    params_.yarn_factor = yarn_factor;
    params_.yarn_beta_fast = yarn_beta_fast;
    params_.yarn_beta_slow = yarn_beta_slow;
    params_.yarn_original_seq_len = yarn_original_seq_len;
    params_.yarn_extrapolation_factor = yarn_extrapolation_factor;

    create_gpu_ropes_if_needed(
        backend_,
        device_,
        params_,
        compress_ratio_,
        model_config,
        config_json,
        rope_theta,
        compress_rope_theta,
        main_rope_,
        compress_rope_);
}

bool DeepseekV4RoPE::use_gpu_forward_() const {
    if (backend_ == Backend::CPU || params_.rope_dim == 0) {
        return false;
    }
    if (!active_gpu_rope_()) {
        return false;
    }
    if (backend_ == Backend::GPU) {
        return true;
    }
    return device_.getType() != infinicore::Device::Type::CPU;
}

const std::shared_ptr<infinicore::nn::RoPE> &DeepseekV4RoPE::active_gpu_rope_() const {
    if (compress_ratio_ > 1) {
        return compress_rope_;
    }
    return main_rope_;
}

infinicore::Tensor DeepseekV4RoPE::forward_cpu_(const infinicore::Tensor &x,
                                                const std::vector<int64_t> &positions) const {
    return apply_rotary_pos_emb(x, positions, params_);
}

infinicore::Tensor DeepseekV4RoPE::prepare_gpu_pos_tensor_(const infinicore::Tensor &pos_ids,
                                                           const infinicore::Device &device) const {
    auto pos_tensor = pos_ids;
    if (!pos_tensor->is_contiguous()) {
        pos_tensor = pos_tensor->contiguous();
    }
    if (pos_tensor->device() != device) {
        pos_tensor = pos_tensor->to(device);
    }
    return pos_tensor;
}

std::tuple<infinicore::Tensor, infinicore::Tensor> DeepseekV4RoPE::forward(const infinicore::Tensor &q,
                                                                           const infinicore::Tensor &k,
                                                                           const infinicore::Tensor &pos_ids) const {
    if (use_gpu_forward_()) {
        const auto pos_tensor = prepare_gpu_pos_tensor_(pos_ids, q->device());
        return {forward_gpu_(q, pos_tensor), forward_gpu_(k, pos_tensor)};
    }
    const auto positions = position_ids_as_vector(pos_ids);
    return {forward_cpu_(q, positions), forward_cpu_(k, positions)};
}

infinicore::Tensor DeepseekV4RoPE::forward_gpu_(const infinicore::Tensor &x,
                                                const infinicore::Tensor &pos_ids) const {
    const auto shape = x->shape();
    if (shape.size() != 4) {
        throw std::runtime_error("DeepseekV4RoPE: forward expects [B,S,H,D]");
    }
    const size_t seq_len = shape[1];
    const size_t head_dim = shape[3];
    const size_t rope_dim = params_.rope_dim;
    if (rope_dim == 0) {
        return x;
    }
    if (pos_ids->shape().size() != 1 || pos_ids->shape()[0] != seq_len) {
        throw std::runtime_error("DeepseekV4RoPE: forward pos_ids length mismatch");
    }
    if (head_dim != params_.head_dim) {
        throw std::runtime_error("DeepseekV4RoPE: forward head_dim mismatch");
    }

    const size_t pass_dim = head_dim - rope_dim;
    auto nope = x->narrow({{3, 0, pass_dim}});
    auto rope_slice = x->narrow({{3, pass_dim, rope_dim}})->contiguous();
    auto rotated = active_gpu_rope_()->forward(rope_slice, pos_ids, true);
    return infinicore::op::cat({nope, rotated}, 3);
}

void DeepseekV4RoPE::forward_blocks(std::vector<float> &kv_comp,
                                    size_t batch_size,
                                    size_t nb,
                                    size_t head_dim,
                                    size_t seq_len,
                                    const std::vector<int64_t> &positions) const {
    if (compress_ratio_ == 0 || nb == 0) {
        return;
    }
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t block = 0; block < nb; ++block) {
            const size_t block_token = std::min(block * compress_ratio_, seq_len > 0 ? seq_len - 1 : 0);
            const int64_t block_pos = (positions[block_token] / static_cast<int64_t>(compress_ratio_))
                                    * static_cast<int64_t>(compress_ratio_);
            const size_t kv_offset = (b * nb + block) * head_dim;
            apply_rope_at_offset(kv_comp, kv_offset, block_pos, params_, false);
        }
    }
}

void DeepseekV4RoPE::inverse_at_offset(std::vector<float> &values,
                                       size_t offset,
                                       int64_t position) const {
    apply_rope_at_offset(values, offset, position, params_, true);
}

} // namespace infinilm::models::deepseek_v4
