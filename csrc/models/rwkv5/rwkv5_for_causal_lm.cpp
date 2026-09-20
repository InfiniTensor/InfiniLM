#include "rwkv5_for_causal_lm.hpp"

#include "../../cache/cache.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/layer_norm.hpp"
#include "infinicore/ops/lerp.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/relu.hpp"
#include "infinicore/ops/rwkv5_wkv.hpp"
#include "infinicore/ops/sigmoid.hpp"
#include "infinicore/ops/silu.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infinilm::models::rwkv5 {
namespace {

std::vector<int32_t> tensor_to_i32_vector(const infinicore::Tensor &tensor,
                                          const char *name) {
    if (!tensor || tensor->dtype() != infinicore::DataType::I32
        || tensor->ndim() != 1) {
        throw std::runtime_error(std::string("RWKV5: ") + name
                                 + " must be a one-dimensional int32 tensor");
    }
    auto cpu = tensor->device() == infinicore::Device::cpu()
                 ? tensor
                 : tensor->to(infinicore::Device::cpu());
    std::vector<int32_t> values(cpu->numel());
    std::memcpy(values.data(), cpu->data(), values.size() * sizeof(int32_t));
    return values;
}

infinicore::Tensor time_mix(const infinicore::Tensor &previous,
                            const infinicore::Tensor &current,
                            const infinicore::Tensor &mix) {
    auto broadcast_mix = mix->as_strided(current->shape(), {0, 0, 1});
    return infinicore::op::lerp(previous, current, broadcast_mix);
}

infinicore::Tensor shift_with_state(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &state_pool,
    const RWKV5BatchMetadata &metadata) {
    const size_t hidden_size = hidden_states->size(2);
    if (hidden_states->ndim() != 3
        || metadata.input_offsets.size() < 2
        || metadata.init_state_indices.size() + 1
               != metadata.input_offsets.size()
        || metadata.final_state_indices.size()
               != metadata.init_state_indices.size()) {
        throw std::runtime_error("RWKV5: invalid shift-state tensor metadata");
    }
    auto previous = infinicore::Tensor::empty(
        hidden_states->shape(), hidden_states->dtype(), hidden_states->device());

    const size_t request_count = metadata.input_offsets.size() - 1;
    if (metadata.input_offsets.back() != static_cast<int32_t>(hidden_states->size(1))) {
        throw std::runtime_error("RWKV5: shift-state offsets do not cover input");
    }
    for (size_t request_idx = 0; request_idx < request_count; ++request_idx) {
        const int32_t start = metadata.input_offsets[request_idx];
        const int32_t end = metadata.input_offsets[request_idx + 1];
        const int32_t read_index = metadata.init_state_indices[request_idx];
        const int32_t write_index = metadata.final_state_indices[request_idx];
        if (start < 0 || end <= start
            || read_index < 0 || write_index < 0
            || static_cast<size_t>(read_index) >= state_pool->size(0)
            || static_cast<size_t>(write_index) >= state_pool->size(0)) {
            throw std::runtime_error("RWKV5: invalid shift-state metadata");
        }

        const size_t token_start = static_cast<size_t>(start);
        const size_t length = static_cast<size_t>(end - start);
        previous->narrow({{1, token_start, 1}})->copy_from(
            state_pool->narrow(
                {{0, static_cast<size_t>(read_index), 1}})
                ->view({1, 1, hidden_size}));
        if (length > 1) {
            previous->narrow({{1, token_start + 1, length - 1}})
                ->copy_from(hidden_states->narrow({{1, token_start, length - 1}}));
        }
        state_pool->narrow({{0, static_cast<size_t>(write_index), 1}})
            ->copy_from(hidden_states->narrow(
                {{1, token_start + length - 1, 1}})
                            ->view({1, hidden_size, 1}));
    }
    return previous;
}

} // namespace

std::shared_ptr<infinilm::config::ModelConfig>
create_rwkv5_model_config(std::shared_ptr<infinilm::config::ModelConfig> config) {
    if (config->get<std::string>("model_type") != "rwkv5") {
        throw std::runtime_error("RWKV5 config creator called for a non-rwkv5 model");
    }

    auto &j = config->get_config_json();
    j["hidden_size"] = j.value("hidden_size", j.value("n_embd", 768));
    j["num_hidden_layers"] = j.value("num_hidden_layers", j.value("n_layer", 12));
    j["intermediate_size"] = j.value(
        "intermediate_size", j.value("n_ffn", 4 * j["hidden_size"].get<size_t>()));
    j["num_attention_heads"] = j.value("num_attention_heads", j.value("n_head", 12));
    j["head_dim"] = j.value(
        "head_dim", j["hidden_size"].get<size_t>() / j["num_attention_heads"].get<size_t>());
    j["layer_norm_eps"] = j.value("layer_norm_eps", 1e-5);
    j["group_norm_eps"] = j.value("group_norm_eps", 64e-5);
    j["use_attention_gate"] = j.value("use_attention_gate", true);
    j["max_position_embeddings"] = j.value(
        "max_position_embeddings", j.value("context_length", 4096));

    const size_t hidden_size = j["hidden_size"].get<size_t>();
    const size_t num_heads = j["num_attention_heads"].get<size_t>();
    const size_t head_dim = j["head_dim"].get<size_t>();
    if (num_heads == 0 || hidden_size != num_heads * head_dim) {
        throw std::runtime_error(
            "RWKV5 requires hidden_size == num_attention_heads * head_dim");
    }
    return config;
}

RWKV5HeadGroupNorm::RWKV5HeadGroupNorm(size_t hidden_size,
                                       size_t num_heads,
                                       double eps,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : num_heads_(num_heads),
      head_size_(hidden_size / num_heads),
      eps_(eps) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({hidden_size}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({hidden_size}, dtype, device));
    unit_weight_ = infinicore::Tensor::ones({head_size_}, dtype, device);
    zero_bias_ = infinicore::Tensor::zeros({head_size_}, dtype, device);
}

infinicore::Tensor RWKV5HeadGroupNorm::forward(
    const infinicore::Tensor &hidden_states) const {
    const size_t token_count = hidden_states->numel() / (num_heads_ * head_size_);
    auto by_head = hidden_states->view({token_count * num_heads_, head_size_});
    auto normalized = infinicore::op::layer_norm(
        by_head, unit_weight_, zero_bias_, static_cast<float>(eps_));
    normalized = normalized->view(hidden_states->shape());
    auto weight = weight_->as_strided(normalized->shape(), {0, 0, 1});
    auto bias = bias_->as_strided(normalized->shape(), {0, 0, 1});
    return infinicore::op::add(infinicore::op::mul(normalized, weight), bias);
}

RWKV5TimeMix::RWKV5TimeMix(
    std::shared_ptr<infinilm::config::ModelConfig> config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = config->get_dtype();
    hidden_size_ = config->get<size_t>("hidden_size");
    num_heads_ = config->get<size_t>("num_attention_heads");
    head_size_ = config->get<size_t>("head_dim");
    use_gate_ = config->get_or<bool>("use_attention_gate", true);
    const double group_norm_eps = config->get<double>("group_norm_eps");

    INFINICORE_NN_PARAMETER_INIT(time_mix_k, ({hidden_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_mix_v, ({hidden_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_mix_r, ({hidden_size_}, dtype, device));
    if (use_gate_) {
        INFINICORE_NN_PARAMETER_INIT(time_mix_g, ({hidden_size_}, dtype, device));
    }
    INFINICORE_NN_PARAMETER_INIT(time_decay, ({num_heads_, head_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_faaaa, ({num_heads_, head_size_}, dtype, device));

    INFINICORE_NN_MODULE_INIT(key, hidden_size_, hidden_size_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(value, hidden_size_, hidden_size_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(receptance, hidden_size_, hidden_size_, false, dtype, device);
    if (use_gate_) {
        INFINICORE_NN_MODULE_INIT(gate, hidden_size_, hidden_size_, false, dtype, device);
    }
    INFINICORE_NN_MODULE_INIT(output, hidden_size_, hidden_size_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        ln_x, hidden_size_, num_heads_, group_norm_eps, dtype, device);
}

infinicore::Tensor RWKV5TimeMix::run_wkv_(
    const infinicore::Tensor &receptance,
    const infinicore::Tensor &key_tensor,
    const infinicore::Tensor &value_tensor,
    const RWKV5BatchMetadata &metadata) const {
    auto &states = infinilm::global_state::get_forward_context().ssm_state_vec;
    if (layer_idx_ >= states.size() || !states[layer_idx_]) {
        throw std::runtime_error("RWKV5TimeMix: WKV state cache is not allocated");
    }
    auto state_pool = states[layer_idx_];
    auto out = infinicore::Tensor::empty(
        receptance->shape(), receptance->dtype(), receptance->device());

    const size_t request_count = metadata.input_offsets.size() - 1;
    for (size_t request_idx = 0; request_idx < request_count; ++request_idx) {
        const int32_t start = metadata.input_offsets[request_idx];
        const int32_t end = metadata.input_offsets[request_idx + 1];
        const int32_t read_index = metadata.init_state_indices[request_idx];
        const int32_t write_index = metadata.final_state_indices[request_idx];
        if (start < 0 || end <= start
            || read_index < 0 || write_index < 0
            || static_cast<size_t>(write_index) >= state_pool->size(0)
            || static_cast<size_t>(read_index) >= state_pool->size(0)) {
            throw std::runtime_error("RWKV5TimeMix: invalid packed sequence metadata");
        }

        const size_t length = static_cast<size_t>(end - start);
        auto read_state = state_pool->narrow(
            {{0, static_cast<size_t>(read_index), 1}});
        infinicore::Tensor request_state;
        if (read_index == write_index) {
            request_state = read_state;
        } else {
            request_state = infinicore::Tensor::empty(
                {1, num_heads_, head_size_, head_size_},
                infinicore::DataType::F32,
                receptance->device());
            request_state->copy_from(read_state);
        }

        infinicore::op::rwkv5_wkv_(
            out->narrow({{1, static_cast<size_t>(start), length}}),
            receptance->narrow({{1, static_cast<size_t>(start), length}}),
            key_tensor->narrow({{1, static_cast<size_t>(start), length}}),
            value_tensor->narrow({{1, static_cast<size_t>(start), length}}),
            time_decay_,
            time_faaaa_,
            request_state);

        if (read_index != write_index) {
            state_pool->narrow({{0, static_cast<size_t>(write_index), 1}})
                ->copy_from(request_state);
        }
    }
    return out;
}

infinicore::Tensor RWKV5TimeMix::forward(
    const infinicore::Tensor &hidden_states,
    const RWKV5BatchMetadata &metadata) const {
    auto &context = infinilm::global_state::get_forward_context();
    const size_t state_idx = layer_idx_ * 2;
    if (state_idx >= context.conv_state_vec.size()
        || !context.conv_state_vec[state_idx]) {
        throw std::runtime_error("RWKV5TimeMix: time-mix state cache is not allocated");
    }
    auto previous = shift_with_state(
        hidden_states, context.conv_state_vec[state_idx], metadata);

    auto k_input = time_mix(previous, hidden_states, time_mix_k_);
    auto v_input = time_mix(previous, hidden_states, time_mix_v_);
    auto r_input = time_mix(previous, hidden_states, time_mix_r_);
    auto k = key_->forward(k_input);
    auto v = value_->forward(v_input);
    auto r = receptance_->forward(r_input);

    auto mixed = ln_x_->forward(run_wkv_(r, k, v, metadata));
    if (use_gate_) {
        auto g_input = time_mix(previous, hidden_states, time_mix_g_);
        auto g = gate_->forward(g_input);
        mixed = infinicore::op::mul(mixed, infinicore::op::silu(g));
    }
    return output_->forward(mixed);
}

RWKV5ChannelMix::RWKV5ChannelMix(
    std::shared_ptr<infinilm::config::ModelConfig> config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t intermediate_size = config->get<size_t>("intermediate_size");
    INFINICORE_NN_PARAMETER_INIT(time_mix_k, ({hidden_size}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_mix_r, ({hidden_size}, dtype, device));
    INFINICORE_NN_MODULE_INIT(key, hidden_size, intermediate_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(value, intermediate_size, hidden_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(receptance, hidden_size, hidden_size, false, dtype, device);
}

infinicore::Tensor RWKV5ChannelMix::forward(
    const infinicore::Tensor &hidden_states,
    const RWKV5BatchMetadata &metadata) const {
    auto &context = infinilm::global_state::get_forward_context();
    const size_t state_idx = layer_idx_ * 2 + 1;
    if (state_idx >= context.conv_state_vec.size()
        || !context.conv_state_vec[state_idx]) {
        throw std::runtime_error("RWKV5ChannelMix: channel-mix state cache is not allocated");
    }
    auto previous = shift_with_state(
        hidden_states, context.conv_state_vec[state_idx], metadata);
    auto k_input = time_mix(previous, hidden_states, time_mix_k_);
    auto r_input = time_mix(previous, hidden_states, time_mix_r_);
    auto k = key_->forward(k_input);
    k = infinicore::op::relu(k);
    k = infinicore::op::mul(k, k);
    auto value = value_->forward(k);
    auto receptance = receptance_->forward(r_input);
    return infinicore::op::mul(infinicore::op::sigmoid(receptance), value);
}

RWKV5Block::RWKV5Block(
    std::shared_ptr<infinilm::config::ModelConfig> config,
    size_t layer_idx,
    const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const double eps = config->get<double>("layer_norm_eps");
    INFINICORE_NN_MODULE_INIT(ln1, hidden_size, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(att, config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(ln2, hidden_size, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(ffn, config, layer_idx, device);
}

infinicore::Tensor RWKV5Block::forward(
    const infinicore::Tensor &hidden_states,
    const RWKV5BatchMetadata &metadata) const {
    auto x = ln1_->forward(hidden_states);
    x = infinicore::op::add(hidden_states, att_->forward(x, metadata));
    auto channel_input = ln2_->forward(x);
    return infinicore::op::add(x, ffn_->forward(channel_input, metadata));
}

RWKV5Model::RWKV5Model(
    std::shared_ptr<infinilm::config::ModelConfig> config,
    const infinicore::Device &device) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.tp_size != 1) {
        throw std::runtime_error("RWKV5 currently supports tensor parallel size 1 only");
    }
    const auto &dtype = config->get_dtype();
    const size_t vocab_size = config->get<size_t>("vocab_size");
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t num_layers = config->get<size_t>("num_hidden_layers");
    const double eps = config->get<double>("layer_norm_eps");
    INFINICORE_NN_MODULE_INIT(
        embeddings, vocab_size, hidden_size, std::nullopt, dtype, device);
    INFINICORE_NN_MODULE_INIT(ln0, hidden_size, eps, dtype, device);
    blocks_.reserve(num_layers);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        blocks_.push_back(this->register_module<RWKV5Block>(
            "blocks." + std::to_string(layer_idx), config, layer_idx, device));
    }
    INFINICORE_NN_MODULE_INIT(ln_out, hidden_size, eps, dtype, device);
}

RWKV5BatchMetadata RWKV5Model::build_batch_metadata_(
    const infinilm::InfinilmModel::Input &input) {
    if (!input.input_offsets || !input.mamba_init_state_indices
        || !input.mamba_final_state_indices) {
        throw std::runtime_error(
            "RWKV5 requires input offsets and initial/final state indices");
    }
    RWKV5BatchMetadata metadata{
        tensor_to_i32_vector(*input.input_offsets, "input_offsets"),
        tensor_to_i32_vector(*input.mamba_init_state_indices, "mamba_init_state_indices"),
        tensor_to_i32_vector(*input.mamba_final_state_indices, "mamba_final_state_indices")};
    if (metadata.input_offsets.size() < 2
        || metadata.init_state_indices.size() + 1 != metadata.input_offsets.size()
        || metadata.final_state_indices.size() != metadata.init_state_indices.size()) {
        throw std::runtime_error("RWKV5 received inconsistent request metadata sizes");
    }
    return metadata;
}

infinicore::Tensor RWKV5Model::forward(
    const infinilm::InfinilmModel::Input &input) const {
    if (!input.input_ids) {
        throw std::runtime_error("RWKV5 requires input_ids");
    }
    const auto metadata = build_batch_metadata_(input);
    auto hidden_states = ln0_->forward(embeddings_->forward(*input.input_ids));
    for (const auto &block : blocks_) {
        hidden_states = block->forward(hidden_states, metadata);
    }
    return ln_out_->forward(hidden_states);
}

RWKV5ForCausalLM::RWKV5ForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> config,
    const infinicore::Device &device)
    : TextCausalLM<RWKV5Model>(std::move(config), device) {}

void RWKV5ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
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
        throw std::runtime_error("RWKV5: unsupported cache configuration");
    }

    const size_t num_layers = model_config_->get<size_t>("num_hidden_layers");
    const size_t hidden_size = model_config_->get<size_t>("hidden_size");
    const size_t num_heads = model_config_->get<size_t>("num_attention_heads");
    const size_t head_size = model_config_->get<size_t>("head_dim");
    const auto &dtype = model_config_->get_dtype();
    const auto device = infinicore::context::getDevice();

    context.conv_state_vec.reserve(num_layers * 2);
    context.ssm_state_vec.reserve(num_layers);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        context.conv_state_vec.push_back(
            infinicore::Tensor::zeros({pool_size, hidden_size, 1}, dtype, device));
        context.conv_state_vec.push_back(
            infinicore::Tensor::zeros({pool_size, hidden_size, 1}, dtype, device));
        context.ssm_state_vec.push_back(infinicore::Tensor::zeros(
            {pool_size, num_heads, head_size, head_size},
            infinicore::DataType::F32,
            device));
    }
    infinicore::context::syncStream();
}

} // namespace infinilm::models::rwkv5

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    rwkv5,
    infinilm::models::rwkv5::RWKV5ForCausalLM,
    infinilm::models::rwkv5::create_rwkv5_model_config);

} // namespace
