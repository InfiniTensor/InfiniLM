#include "granitemoehybrid_sparse_moe_block.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mul_scalar.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::granitemoehybrid {

GraniteMoeHybridExpertMLP::GraniteMoeHybridExpertMLP(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("intermediate_size");
    const bool use_bias = model_config->get_or<bool>("mlp_bias", false);
    const std::string hidden_act =
        model_config->get_or<std::string>("hidden_act", "silu");

    const auto &dtype = model_config->get_dtype();
    const engine::distributed::RankInfo &rank_info =
        infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.tp_size <= 0 ||
        intermediate_size % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridExpertMLP: "
            "intermediate_size must be divisible by tp_size");
    }

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn =
        [this](const std::string &name, infinicore::nn::Parameter parameter) {
            this->register_parameter(name, std::move(parameter));
        };
    input_linear_ =
        std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
            hidden_size,
            intermediate_size,
            "input_linear.gate",
            "input_linear.up",
            register_fn,
            quantization_method,
            use_bias,
            dtype,
            device,
            rank_info);
    output_linear_ =
        this->register_module<infinilm::layers::linear::RowParallelLinear>(
            "output_linear",
            intermediate_size,
            hidden_size,
            quantization_method,
            use_bias,
            dtype,
            device,
            rank_info.tp_rank,
            rank_info.tp_size,
            rank_info.comm);
}

infinicore::Tensor GraniteMoeHybridExpertMLP::forward(
    const infinicore::Tensor &hidden_states) const {
    auto input = hidden_states;
    auto gate_up = input_linear_->forward(input);
    auto activated = infinicore::op::silu_and_mul(gate_up);
    return output_linear_->forward(activated);
}

GraniteMoeHybridExperts::GraniteMoeHybridExperts(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    num_experts_ = model_config->get<size_t>("num_local_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    residual_multiplier_ =
        model_config->get_or<float>("residual_multiplier", 1.0f);

    if (num_experts_ == 0 || num_experts_per_tok_ == 0
        || num_experts_per_tok_ > num_experts_) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridExperts: "
            "num_experts_per_tok must be in [1, num_local_experts]");
    }

    experts_.reserve(num_experts_);
    for (size_t expert = 0; expert < num_experts_; ++expert) {
        experts_.push_back(
            this->register_module<GraniteMoeHybridExpertMLP>(
                std::to_string(expert), model_config, device));
    }
}

infinicore::Tensor GraniteMoeHybridExperts::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &selected_experts,
    const infinicore::Tensor &routing_weights) const {
    if (hidden_states->ndim() != 2) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridExperts::forward: "
            "hidden_states must have shape [num_tokens, hidden_size]");
    }

    auto selected_experts_cpu =
        selected_experts->to(infinicore::Device::Type::CPU);
    auto routing_weights_cpu =
        routing_weights->to(infinicore::Device::Type::CPU);
    const auto *selected_experts_ptr =
        reinterpret_cast<const int *>(selected_experts_cpu->data());
    const auto *routing_weights_ptr =
        reinterpret_cast<const float *>(routing_weights_cpu->data());

    const size_t num_tokens = hidden_states->shape()[0];
    auto output = infinicore::Tensor::empty(
        hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    for (size_t token = 0; token < num_tokens; ++token) {
        auto token_input = hidden_states->narrow({{0, token, 1}});
        const size_t route_offset = token * num_experts_per_tok_;
        infinicore::Tensor token_output;

        for (size_t route = 0; route < num_experts_per_tok_; ++route) {
            const int expert = selected_experts_ptr[route_offset + route];
            if (expert < 0 || static_cast<size_t>(expert) >= num_experts_) {
                throw std::runtime_error(
                    "infinilm::models::granitemoehybrid::GraniteMoeHybridExperts::forward: "
                    "router selected an invalid expert index");
            }

            const float scale =
                routing_weights_ptr[route_offset + route]
                * residual_multiplier_;
            auto expert_output = experts_[expert]->forward(token_input);
            expert_output = infinicore::op::mul_scalar(expert_output, scale);
            if (route == 0) {
                token_output = expert_output;
            } else {
                infinicore::op::add_(
                    token_output, token_output, expert_output);
            }
        }
        output->narrow({{0, token, 1}})->copy_from(token_output);
    }
    return output;
}

GraniteMoeHybridTopKRouter::GraniteMoeHybridTopKRouter(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_experts = model_config->get<size_t>("num_local_experts");
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");

    if (num_experts == 0 || num_experts_per_tok_ == 0
        || num_experts_per_tok_ > num_experts) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridTopKRouter: "
            "num_experts_per_tok must be in [1, num_local_experts]");
    }

    INFINICORE_NN_MODULE_INIT(
        layer, hidden_size, num_experts, false, dtype, device);
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
GraniteMoeHybridTopKRouter::forward(
    const infinicore::Tensor &hidden_states) const {
    if (hidden_states->ndim() != 2) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridTopKRouter::forward: "
            "hidden_states must have shape [num_tokens, hidden_size]");
    }

    const size_t num_tokens = hidden_states->shape()[0];
    auto input = hidden_states;
    auto router_logits = layer_->forward(input);
    auto router_scores = infinicore::Tensor::empty(
        {num_tokens, num_experts_per_tok_},
        infinicore::DataType::F32,
        hidden_states->device());
    auto router_indices = infinicore::Tensor::empty(
        {num_tokens, num_experts_per_tok_},
        infinicore::DataType::I32,
        hidden_states->device());

    infinicore::op::topksoftmax(
        router_scores,
        router_indices,
        router_logits,
        num_experts_per_tok_,
        true);
    return {router_scores, router_indices};
}

GraniteMoeHybridSparseMoeBlock::GraniteMoeHybridSparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(router, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
}

infinicore::Tensor GraniteMoeHybridSparseMoeBlock::forward(
    const infinicore::Tensor &hidden_states) const {
    if (hidden_states->ndim() != 3) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridSparseMoeBlock::forward: "
            "hidden_states must have shape [batch_size, sequence_length, hidden_size]");
    }

    const auto &shape = hidden_states->shape();
    auto flat_hidden_states = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto [routing_weights, selected_experts] = router_->forward(flat_hidden_states);
    auto output = experts_->forward(
        flat_hidden_states, selected_experts, routing_weights);
    return output->view(shape);
}

} // namespace infinilm::models::granitemoehybrid
