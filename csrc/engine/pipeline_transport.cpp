#include "pipeline_transport.hpp"

#include <infinicore/context/context.hpp>

#include <stdexcept>
#include <utility>

namespace infinilm::engine {
namespace {

void validate_activation(
    const PipelineActivation &source,
    const infinicore::Device &destination) {
    if (!source.hidden_states || !source.residual) {
        throw std::invalid_argument(
            "Pipeline peer transport requires hidden states and residual");
    }

    const auto source_device = source.hidden_states->device();
    if (source.residual->device() != source_device) {
        throw std::invalid_argument(
            "Pipeline activation tensors must be on the same source device");
    }
    if (source_device.getType() != infinicore::Device::Type::NVIDIA
        || destination.getType() != infinicore::Device::Type::NVIDIA) {
        throw std::invalid_argument(
            "Pipeline peer transport currently supports NVIDIA devices only");
    }
    if (source_device == destination) {
        throw std::invalid_argument(
            "Pipeline source and destination devices must be distinct");
    }
    if (!source.hidden_states->is_contiguous()
        || !source.residual->is_contiguous()) {
        throw std::invalid_argument(
            "Pipeline activation tensors must be contiguous");
    }
    if (source.hidden_states->shape() != source.residual->shape()
        || source.hidden_states->dtype() != source.residual->dtype()) {
        throw std::invalid_argument(
            "Pipeline hidden states and residual must have matching shape and dtype");
    }
}

} // namespace

PipelineActivation PeerCopyTransport::transfer(
    const PipelineActivation &source,
    const infinicore::Device &destination) {
    validate_activation(source, destination);

    const auto previous_device = infinicore::context::getDevice();
    try {
        auto hidden_states = source.hidden_states->to(destination);
        auto residual = source.residual->to(destination);

        // Tensor::to performs a synchronous peer hand-off in the MVP. Keep an
        // explicit destination sync as part of this transport's public contract.
        infinicore::context::setDevice(destination);
        infinicore::context::syncStream();
        infinicore::context::setDevice(previous_device);

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++transfers_;
            tensors_ += 2;
            bytes_ += hidden_states->nbytes() + residual->nbytes();
        }
        return {std::move(hidden_states), std::move(residual)};
    } catch (...) {
        try {
            infinicore::context::setDevice(previous_device);
        } catch (...) {
        }
        throw;
    }
}

PipelineTransportStats PeerCopyTransport::stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return {
        "gpu-peer-copy",
        transfers_,
        tensors_,
        bytes_};
}

} // namespace infinilm::engine
