#include "pipeline_transport.hpp"

#include <infinicore/context/context.hpp>
#include <infinicore/device_event.hpp>

#include <memory>
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

class PipelineTransferState {
public:
    PipelineTransferState(
        PipelineActivation source,
        std::shared_ptr<infinicore::DeviceEvent> source_ready,
        std::shared_ptr<infinicore::DeviceEvent> copy_done)
        : source_(std::move(source)),
          source_ready_(std::move(source_ready)),
          copy_done_(std::move(copy_done)) {}

    ~PipelineTransferState() {
        try {
            if (copy_done_ && !copy_done_->query()) {
                copy_done_->synchronize();
            }
        } catch (...) {
        }
    }

    bool complete() const {
        return copy_done_ == nullptr || copy_done_->query();
    }

    void wait() const {
        if (copy_done_) {
            copy_done_->synchronize();
        }
    }

private:
    PipelineActivation source_;
    std::shared_ptr<infinicore::DeviceEvent> source_ready_;
    std::shared_ptr<infinicore::DeviceEvent> copy_done_;
};

PipelineTransfer::PipelineTransfer(
    PipelineActivation activation,
    std::shared_ptr<PipelineTransferState> state)
    : activation_(std::move(activation)), state_(std::move(state)) {}

const PipelineActivation &PipelineTransfer::activation() const {
    return activation_;
}

bool PipelineTransfer::complete() const {
    return state_ == nullptr || state_->complete();
}

void PipelineTransfer::wait() const {
    if (state_) {
        state_->wait();
    }
}

PipelineTransfer PeerCopyTransport::transfer_async(
    const PipelineActivation &source,
    const infinicore::Device &destination) {
    validate_activation(source, destination);

    const auto previous_device = infinicore::context::getDevice();
    const auto source_device = source.hidden_states->device();
    infinicore::Tensor hidden_states;
    infinicore::Tensor residual;
    std::shared_ptr<infinicore::DeviceEvent> source_ready;
    std::shared_ptr<infinicore::DeviceEvent> copy_done;
    bool destination_work_may_be_enqueued = false;
    try {
        source_ready = std::make_shared<infinicore::DeviceEvent>(
            source_device, INFINIRT_EVENT_DISABLE_TIMING);
        source_ready->record();

        infinicore::context::setDevice(destination);
        hidden_states = infinicore::Tensor::empty(
            source.hidden_states->shape(), source.hidden_states->dtype(), destination);
        residual = infinicore::Tensor::empty(
            source.residual->shape(), source.residual->dtype(), destination);

        const auto destination_stream = infinicore::context::getStream();
        destination_work_may_be_enqueued = true;
        source_ready->wait_on(destination, destination_stream);
        infinicore::context::memcpyPeerD2D(
            hidden_states->data(),
            source.hidden_states->data(),
            source_device,
            hidden_states->nbytes());
        infinicore::context::memcpyPeerD2D(
            residual->data(),
            source.residual->data(),
            source_device,
            residual->nbytes());

        copy_done = std::make_shared<infinicore::DeviceEvent>(
            destination, INFINIRT_EVENT_DISABLE_TIMING);
        copy_done->record(destination_stream);
        infinicore::context::setDevice(previous_device);

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++transfers_;
            tensors_ += 2;
            bytes_ += hidden_states->nbytes() + residual->nbytes();
        }

        auto state = std::make_shared<PipelineTransferState>(
            PipelineActivation{source.hidden_states, source.residual},
            std::move(source_ready),
            std::move(copy_done));
        PipelineActivation destination_activation{
            std::move(hidden_states), std::move(residual)};
        return PipelineTransfer(
            std::move(destination_activation), std::move(state));
    } catch (...) {
        if (destination_work_may_be_enqueued) {
            try {
                infinicore::context::setDevice(destination);
                infinicore::context::syncStream();
            } catch (...) {
            }
        }
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
        "gpu-peer-copy-async-event",
        transfers_,
        tensors_,
        bytes_};
}

} // namespace infinilm::engine
