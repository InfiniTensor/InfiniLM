#include "mamba2_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/causal_conv1d.hpp"
#include "infinicore/ops/mamba_selective_scan.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/silu.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infinilm::models::mamba2 {
namespace {

std::vector<int32_t> tensor_to_i32_vector(const infinicore::Tensor &tensor,
                                          const char *name) {
    if (!tensor || tensor->dtype() != infinicore::DataType::I32 || tensor->ndim() != 1) {
        throw std::runtime_error(std::string("Mamba2: ") + name
                                 + " must be a one-dimensional int32 tensor");
    }
    auto cpu = tensor->device() == infinicore::Device::cpu()
                 ? tensor
                 : tensor->to(infinicore::Device::cpu());
    std::vector<int32_t> values(cpu->numel());
    std::memcpy(values.data(), cpu->data(), values.size() * sizeof(int32_t));
    return values;
}

} // namespace

std::shared_ptr<infinilm::config::ModelConfig>
create_mamba2_model_config(std::shared_ptr<infinilm::config::ModelConfig> config) {
    if (config->get<std::string>("model_type") != "mamba2") {
        throw std::runtime_error("Mamba2 config creator called for a non-mamba2 model");
    }
    auto &j = config->get_config_json();
    j["hidden_size"] = j.value("hidden_size", j.value("d_model", 768));
    j["num_hidden_layers"] = j.value("num_hidden_layers", j.value("n_layer", 24));
    j["intermediate_size"] = j.value(
        "intermediate_size", j.value("d_inner", j["hidden_size"].get<size_t>() * j.value("expand", 2)));
    j["state_size"] = j.value("state_size", j.value("ssm_state_size", j.value("d_state", 64)));
    j["conv_kernel"] = j.value("conv_kernel", j.value("d_conv", 4));
    j["num_heads"] = j.value("num_heads", j.value("nheads", 0));
    j["head_dim"] = j.value("head_dim", 0);
    if (j["num_heads"].get<size_t>() == 0 && j["head_dim"].get<size_t>() == 0) {
        j["head_dim"] = 64;
        j["num_heads"] = j["intermediate_size"].get<size_t>() / 64;
    } else if (j["num_heads"].get<size_t>() == 0) {
        j["num_heads"] = j["intermediate_size"].get<size_t>() / j["head_dim"].get<size_t>();
    } else if (j["head_dim"].get<size_t>() == 0) {
        j["head_dim"] = j["intermediate_size"].get<size_t>() / j["num_heads"].get<size_t>();
    }
    j["num_groups"] = j.value("num_groups", j.value("n_groups", 1));
    j["layer_norm_epsilon"] = j.value("layer_norm_epsilon", j.value("rms_norm_eps", 1e-5));
    j["rms_norm_eps"] = j.value("rms_norm_eps", j["layer_norm_epsilon"]);
    j["norm_dim"] = j.value("norm_dim", j.value("mamba2_norm_dim", j["head_dim"]));
    j["use_bias"] = j.value("use_bias", false);
    j["use_conv_bias"] = j.value("use_conv_bias", true);
    j["max_position_embeddings"] = j.value("max_position_embeddings", 8192);

    const size_t intermediate = j["intermediate_size"].get<size_t>();
    const size_t heads = j["num_heads"].get<size_t>();
    const size_t head_dim = j["head_dim"].get<size_t>();
    const size_t groups = j["num_groups"].get<size_t>();
    if (heads == 0 || head_dim == 0 || intermediate != heads * head_dim) {
        throw std::runtime_error("Mamba2 requires intermediate_size == num_heads * head_dim");
    }
    if (groups == 0 || heads % groups != 0) {
        throw std::runtime_error("Mamba2 requires num_heads divisible by num_groups");
    }
    if (groups != 1) {
        throw std::runtime_error("Mamba2 adapter currently supports num_groups=1");
    }
    return config;
}

Mamba2Mixer::Mamba2Mixer(std::shared_ptr<infinilm::config::ModelConfig> config,
                         size_t layer_idx,
                         const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = config->get_dtype();
    hidden_size_ = config->get<size_t>("hidden_size");
    intermediate_size_ = config->get<size_t>("intermediate_size");
    state_size_ = config->get<size_t>("state_size");
    num_heads_ = config->get<size_t>("num_heads");
    head_dim_ = config->get<size_t>("head_dim");
    const size_t norm_dim = config->get_or<size_t>("norm_dim", head_dim_);
    num_groups_ = config->get<size_t>("num_groups");
    conv_kernel_ = config->get<size_t>("conv_kernel");
    conv_dim_ = intermediate_size_ + 2 * num_groups_ * state_size_;

    const bool use_bias = config->get_or<bool>("use_bias", false);
    const bool use_conv_bias = config->get_or<bool>("use_conv_bias", true);
    const size_t projection_size = intermediate_size_ + conv_dim_ + num_heads_;
    INFINICORE_NN_MODULE_INIT(in_proj, hidden_size_, projection_size, use_bias, dtype, device);
    INFINICORE_NN_MODULE_INIT(out_proj, intermediate_size_, hidden_size_, use_bias, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, norm_dim, config->get<double>("rms_norm_eps"), dtype, device);
    INFINICORE_NN_PARAMETER_INIT(conv1d_weight, ({conv_dim_, 1, conv_kernel_}, dtype, device));
    if (use_conv_bias) {
        INFINICORE_NN_PARAMETER_INIT(conv1d_bias, ({conv_dim_}, dtype, device));
    }
    INFINICORE_NN_PARAMETER_INIT(A_log, ({intermediate_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(D, ({intermediate_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(dt_bias, ({intermediate_size_}, dtype, device));
}

infinicore::Tensor Mamba2Mixer::forward(
    const infinicore::Tensor &hidden_states,
    const Mamba2BatchMetadata &metadata) const {
    auto hidden_mut = const_cast<infinicore::Tensor &>(hidden_states);
    auto projected = in_proj_->forward(hidden_mut);
    auto z = projected->narrow({{2, 0, intermediate_size_}})->contiguous();
    auto xbc = projected->narrow({{2, intermediate_size_, conv_dim_}})->contiguous();
    auto dt = projected->narrow({{2, intermediate_size_ + conv_dim_, num_heads_}})->contiguous();

    auto &context = infinilm::global_state::get_forward_context();
    if (layer_idx_ >= context.conv_state_vec.size() || !context.conv_state_vec[layer_idx_]
        || layer_idx_ >= context.ssm_state_vec.size() || !context.ssm_state_vec[layer_idx_]) {
        throw std::runtime_error("Mamba2 mixer state cache is not allocated");
    }
    if (!context.mamba_metadata.input_offsets
        || !context.mamba_metadata.init_state_indices
        || !context.mamba_metadata.final_state_indices) {
        throw std::runtime_error("Mamba2 mixer requires state metadata");
    }
    auto conv_out = infinicore::op::causal_conv1d(
        xbc,
        context.conv_state_vec[layer_idx_],
        conv1d_weight_,
        conv1d_bias_ ? std::optional<infinicore::Tensor>(conv1d_bias_) : std::nullopt,
        context.mamba_metadata.input_offsets,
        context.mamba_metadata.init_state_indices,
        context.mamba_metadata.final_state_indices);
    conv_out = infinicore::op::silu(conv_out);
    auto x = conv_out->narrow({{2, 0, intermediate_size_}})->contiguous();
    auto b = conv_out->narrow({{2, intermediate_size_, state_size_}})->contiguous();
    auto c = conv_out->narrow({{2, intermediate_size_ + state_size_, state_size_}})->contiguous();

    auto ssm_output = infinicore::Tensor::empty(
        {hidden_states->size(0), hidden_states->size(1), intermediate_size_},
        hidden_states->dtype(), hidden_states->device());
    auto state_pool = context.ssm_state_vec[layer_idx_];
    const size_t request_count = metadata.input_offsets.size() - 1;
    if (metadata.input_offsets.back() != static_cast<int32_t>(hidden_states->size(1))) {
        throw std::runtime_error("Mamba2 input offsets do not cover hidden states");
    }
    auto a_log_scan = infinicore::op::broadcast_to(
                           A_log_->view({intermediate_size_, 1}),
                           {intermediate_size_, state_size_})
                           ->contiguous();
    auto d_scan = D_->view({intermediate_size_});
    auto dt_bias_scan = dt_bias_->view({intermediate_size_});

    for (size_t request_idx = 0; request_idx < request_count; ++request_idx) {
        const int32_t start = metadata.input_offsets[request_idx];
        const int32_t end = metadata.input_offsets[request_idx + 1];
        const int32_t read_index = metadata.init_state_indices[request_idx];
        const int32_t write_index = metadata.final_state_indices[request_idx];
        if (start < 0 || end <= start || read_index < 0 || write_index < 0
            || static_cast<size_t>(read_index) >= state_pool->size(0)
            || static_cast<size_t>(write_index) >= state_pool->size(0)) {
            throw std::runtime_error("Mamba2 received invalid state metadata");
        }
        const size_t token_start = static_cast<size_t>(start);
        const size_t length = static_cast<size_t>(end - start);
        auto read_state = state_pool->narrow({{0, static_cast<size_t>(read_index), 1}});
        infinicore::Tensor request_state;
        if (read_index == write_index) {
            request_state = read_state->view({1, intermediate_size_, state_size_});
        } else {
            request_state = infinicore::Tensor::empty(
                {1, intermediate_size_, state_size_}, infinicore::DataType::F32,
                hidden_states->device());
            request_state->copy_from(read_state->view({1, intermediate_size_, state_size_}));
        }
        auto request_dt = dt->narrow({{1, token_start, length}})->view({1, length, num_heads_, 1});
        request_dt = infinicore::op::broadcast_to(
                         request_dt, {1, length, num_heads_, head_dim_})
                         ->contiguous()
                         ->view({1, length, intermediate_size_});
        // The shared scan operator applies silu(gate) internally. Its neutral
        // input is the positive solution of silu(x) = 1; Mamba2 applies its
        // actual gate after per-head RMSNorm below.
        auto neutral_gate = infinicore::op::mul_scalar(
            infinicore::Tensor::ones(
                x->narrow({{1, token_start, length}})->shape(),
                x->dtype(),
                x->device()),
            1.2784645427610737);
        auto scan = infinicore::op::mamba_selective_scan(
            x->narrow({{1, token_start, length}}), request_dt,
            b->narrow({{1, token_start, length}}),
            c->narrow({{1, token_start, length}}),
            a_log_scan, d_scan, neutral_gate,
            dt_bias_scan, request_state);
        const size_t norm_dim = norm_->weight()->size(0);
        infinicore::Tensor normalized;
        if (norm_dim == intermediate_size_) {
            normalized = norm_->forward(scan->view({length, intermediate_size_}));
            normalized = normalized->view({1, length, intermediate_size_});
        } else if (norm_dim == head_dim_) {
            normalized = norm_->forward(scan->view({length * num_heads_, head_dim_}));
            normalized = normalized->view({1, length, intermediate_size_});
        } else {
            throw std::runtime_error("Mamba2 norm_dim must equal intermediate_size or head_dim");
        }
        normalized = infinicore::op::mul(
            normalized,
            infinicore::op::silu(z->narrow({{1, token_start, length}})));
        ssm_output->narrow({{1, token_start, length}})->copy_from(normalized);
        if (read_index != write_index) {
            state_pool->narrow({{0, static_cast<size_t>(write_index), 1}})
                ->copy_from(request_state->view({1, num_heads_, head_dim_, state_size_}));
        }
    }
    auto output_mut = ssm_output;
    return out_proj_->forward(output_mut);
}

Mamba2Block::Mamba2Block(std::shared_ptr<infinilm::config::ModelConfig> config,
                         size_t layer_idx,
                         const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, config->get<double>("rms_norm_eps"), dtype, device);
    INFINICORE_NN_MODULE_INIT(mixer, config, layer_idx, device);
}

infinicore::Tensor Mamba2Block::forward(
    const infinicore::Tensor &hidden_states,
    const Mamba2BatchMetadata &metadata) const {
    auto normalized = norm_->forward(hidden_states);
    return infinicore::op::add(hidden_states, mixer_->forward(normalized, metadata));
}

Mamba2Model::Mamba2Model(std::shared_ptr<infinilm::config::ModelConfig> config,
                         const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t vocab_size = config->get<size_t>("vocab_size");
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t num_layers = config->get<size_t>("num_hidden_layers");
    INFINICORE_NN_MODULE_INIT(embedding, vocab_size, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_layers);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        layers_.push_back(this->register_module<Mamba2Block>(
            "layers." + std::to_string(layer_idx), config, layer_idx, device));
    }
    INFINICORE_NN_MODULE_INIT(norm_f, hidden_size, config->get<double>("rms_norm_eps"), dtype, device);
}

Mamba2BatchMetadata Mamba2Model::build_batch_metadata_(
    const infinilm::InfinilmModel::Input &input) {
    if (!input.input_offsets || !input.mamba_init_state_indices
        || !input.mamba_final_state_indices) {
        throw std::runtime_error("Mamba2 requires input offsets and state indices");
    }
    Mamba2BatchMetadata metadata{
        tensor_to_i32_vector(*input.input_offsets, "input_offsets"),
        tensor_to_i32_vector(*input.mamba_init_state_indices, "mamba_init_state_indices"),
        tensor_to_i32_vector(*input.mamba_final_state_indices, "mamba_final_state_indices")};
    if (metadata.input_offsets.size() < 2
        || metadata.init_state_indices.size() + 1 != metadata.input_offsets.size()
        || metadata.final_state_indices.size() != metadata.init_state_indices.size()) {
        throw std::runtime_error("Mamba2 received inconsistent request metadata sizes");
    }
    return metadata;
}

infinicore::Tensor Mamba2Model::forward(
    const infinilm::InfinilmModel::Input &input) const {
    if (!input.input_ids) {
        throw std::runtime_error("Mamba2 requires input_ids");
    }
    const auto metadata = build_batch_metadata_(input);
    auto input_ids = input.input_ids.value();
    if (input_ids->ndim() == 1) {
        input_ids = input_ids->view({1, input_ids->size(0)});
    }
    auto hidden_states = embedding_->forward(input_ids);
    for (const auto &layer : layers_) {
        hidden_states = layer->forward(hidden_states, metadata);
    }
    return norm_f_->forward(hidden_states);
}

Mamba2ForCausalLM::Mamba2ForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> config,
    const infinicore::Device &device)
    : infinilm::layers::causal_lm_templates::TextCausalLM<Mamba2Model>(
          std::move(config), device) {}

void Mamba2ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    auto &context = infinilm::global_state::get_forward_context();
    context.kv_cache_vec.clear();
    context.conv_state_vec.clear();
    context.ssm_state_vec.clear();
    if (cache_config == nullptr) {
        cache_config_.reset();
        return;
    }
    cache_config_ = cache_config->unique_copy();
    size_t pool_size = 0;
    if (const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        pool_size = std::max<size_t>(2, paged->num_blocks() / 4);
    } else if (const auto *fixed = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        pool_size = fixed->max_batch_size() + 1;
    } else {
        throw std::runtime_error("Mamba2: unsupported cache configuration");
    }
    const size_t num_layers = model_config_->get<size_t>("num_hidden_layers");
    const size_t conv_dim = model_config_->get<size_t>("intermediate_size")
                            + 2 * model_config_->get<size_t>("num_groups")
                                  * model_config_->get<size_t>("state_size");
    const size_t conv_kernel = model_config_->get<size_t>("conv_kernel");
    const size_t num_heads = model_config_->get<size_t>("num_heads");
    const size_t head_dim = model_config_->get<size_t>("head_dim");
    const size_t state_size = model_config_->get<size_t>("state_size");
    const auto &dtype = model_config_->get_dtype();
    const auto device = infinicore::context::getDevice();
    context.conv_state_vec.reserve(num_layers);
    context.ssm_state_vec.reserve(num_layers);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        context.conv_state_vec.push_back(infinicore::Tensor::zeros(
            {pool_size, conv_dim, conv_kernel - 1}, dtype, device));
        context.ssm_state_vec.push_back(infinicore::Tensor::zeros(
            {pool_size, num_heads, head_dim, state_size},
            infinicore::DataType::F32, device));
    }
    infinicore::context::syncStream();
}

} // namespace infinilm::models::mamba2

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    mamba2,
    infinilm::models::mamba2::Mamba2ForCausalLM,
    infinilm::models::mamba2::create_mamba2_model_config);
} // namespace
