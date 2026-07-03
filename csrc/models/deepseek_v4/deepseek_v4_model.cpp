#include "deepseek_v4_model.hpp"

#include "deepseek_v4_utils.hpp"
#include "infinicore/context/context.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace infinilm::models::deepseek_v4 {
namespace {

infinicore::Tensor embedding_reference(const infinicore::Tensor &indices,
                                       const infinicore::Tensor &weight,
                                       size_t vocab_size,
                                       size_t hidden_size) {
    const auto ids = tensor_to_int64_vector(indices);
    auto output_shape = indices->shape();
    output_shape.push_back(hidden_size);

    size_t num_lookups = 1;
    for (auto dim : indices->shape()) {
        num_lookups *= dim;
    }
    if (ids.size() != num_lookups) {
        throw std::runtime_error("DeepseekV4Model embedding_reference: id count mismatch");
    }

    auto output = infinicore::Tensor::empty(output_shape, weight->dtype(), weight->device());
    auto flat_output = output->view({num_lookups, hidden_size});
    for (size_t i = 0; i < num_lookups; ++i) {
        const int64_t token_id = ids[i];
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
            throw std::out_of_range("DeepseekV4Model embedding token id out of range: " + std::to_string(token_id));
        }
        auto dst = flat_output->narrow({{0, i, 1}});
        auto src = weight->narrow({{0, static_cast<size_t>(token_id), 1}});
        dst->copy_from(src);
    }
    if (output->device().getType() != infinicore::Device::Type::CPU) {
        infinicore::context::syncStream();
    }
    return output;
}

} // namespace

DeepseekV4Model::DeepseekV4Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      vocab_size_(model_config->get<size_t>("vocab_size")),
      hc_mult_(model_config->get<size_t>("hc_mult")),
      hc_eps_(model_config->get<double>("hc_eps")) {
    const auto &dtype = model_config->get_dtype();
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed, vocab_size_, hidden_size_, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<DeepseekV4DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, rms_norm_eps, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(hc_head_base, ({hc_mult_}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_head_fn, ({hc_mult_, hidden_size_ * hc_mult_}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_head_scale, ({static_cast<size_t>(1)}, infinicore::DataType::F32, device));
}

infinicore::Tensor DeepseekV4Model::forward(const infinicore::Tensor &input_ids,
                                            const infinicore::Tensor &positions) const {
    auto hidden_states = embedding_reference(input_ids, embed_->weight(), vocab_size_, hidden_size_);
    hidden_states = expand_hc_stream(hidden_states, hc_mult_);

    infinicore::Tensor residual;
    infinicore::Tensor post_mix;
    infinicore::Tensor res_mix;
    for (const auto &layer : layers_) {
        auto layer_output = layer->forward(hidden_states, positions, input_ids, post_mix, res_mix, residual);
        hidden_states = std::get<0>(layer_output);
        residual = std::get<1>(layer_output);
        post_mix = std::get<2>(layer_output);
        res_mix = std::get<3>(layer_output);
    }

    hidden_states = mhc_head_pre(hidden_states, hc_head_base_, hc_head_fn_, hc_head_scale_,
                                 hc_mult_, hidden_size_, hc_eps_);
    debug_trace_tensor("20_model_hc_head", hidden_states);
    auto final_hidden = norm_->forward(hidden_states);
    debug_trace_tensor("21_model_final_norm", final_hidden);
    return final_hidden;
}

} // namespace infinilm::models::deepseek_v4
