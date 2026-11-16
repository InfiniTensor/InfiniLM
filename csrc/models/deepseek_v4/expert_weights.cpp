#include "expert_weights.hpp"

#include <limits>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {

W8A8ExpertWeights::W8A8ExpertWeights(runtime::Runtime &rt, size_t experts, size_t hidden,
                                     size_t intermediate, size_t rank, size_t ranks, bool shared) {
    using runtime::DataType;
    using runtime::Tensor;
    using weights::Weight;
    if (!experts || !hidden || !intermediate || !ranks || rank >= ranks
        || intermediate % ranks || (shared && experts != 1)) {
        throw std::invalid_argument("Invalid DeepSeek V4 W8A8 expert TP geometry");
    }
    const auto local = intermediate / ranks;
    const auto limit = std::numeric_limits<size_t>::max();
    if (local > limit / 2 || experts > limit / (2 * local) / hidden
        || experts > limit / hidden / sizeof(float)
        || experts > limit / (2 * local) / sizeof(float)) {
        throw std::overflow_error("DeepSeek V4 W8A8 expert weight size overflows");
    }

    gate_up_ = Tensor::empty_parameter(rt, {experts, 2 * local, hidden}, DataType::kInt8);
    gate_up_scale_ = Tensor::empty_parameter(rt, {experts, 2 * local, 1}, DataType::kFloat32);
    down_ = Tensor::empty_parameter(rt, {experts, hidden, local}, DataType::kInt8);
    down_scale_ = Tensor::empty_parameter(rt, {experts, hidden, 1}, DataType::kFloat32);

    for (size_t expert = 0; expert < experts; ++expert) {
        const auto prefix = shared ? std::string{} : std::to_string(expert) + ".";
        const auto select_gate = [&](const Tensor &pack, size_t offset) {
            return pack->narrow({{0, expert, 1}, {1, offset, local}})->squeeze(0);
        };
        bind_weight(prefix + "w1.weight", Weight(select_gate(gate_up_, 0), 0, rank, ranks));
        bind_weight(prefix + "w3.weight", Weight(select_gate(gate_up_, local), 0, rank, ranks));
        bind_weight(prefix + "w1.weight_scale", Weight(select_gate(gate_up_scale_, 0), 0, rank, ranks));
        bind_weight(prefix + "w3.weight_scale", Weight(select_gate(gate_up_scale_, local), 0, rank, ranks));
        bind_weight(prefix + "w2.weight", Weight(down_->narrow({{0, expert, 1}})->squeeze(0), 1, rank, ranks));
        bind_weight(prefix + "w2.weight_scale", Weight(down_scale_->narrow({{0, expert, 1}})->squeeze(0)));
    }
}

} // namespace infinilm::models::deepseek_v4
