#pragma once

#include <infinicore/device.hpp>
#include <infinicore/tensor.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>

namespace infinilm::engine {

struct PipelineActivation {
    infinicore::Tensor hidden_states;
    infinicore::Tensor residual;
};

struct PipelineTransportStats {
    std::string name{"none"};
    size_t transfers{0};
    size_t tensors{0};
    size_t bytes{0};
};

class PipelineTransferState;

// A transfer ticket is also the activation-buffer lease. It retains the
// producer tensors until copy completion and the consumer tensors until the
// ticket is released.
class PipelineTransfer {
public:
    PipelineTransfer() = default;
    PipelineTransfer(
        PipelineActivation activation,
        std::shared_ptr<PipelineTransferState> state);

    const PipelineActivation &activation() const;
    bool complete() const;
    void wait() const;

private:
    PipelineActivation activation_;
    std::shared_ptr<PipelineTransferState> state_;
};

class PipelineTransport {
public:
    virtual ~PipelineTransport() = default;

    // Enqueue an event-ordered transfer without synchronizing either device.
    virtual PipelineTransfer transfer_async(
        const PipelineActivation &source,
        const infinicore::Device &destination) = 0;

    virtual PipelineTransportStats stats() const = 0;
};

class PeerCopyTransport final : public PipelineTransport {
public:
    PipelineTransfer transfer_async(
        const PipelineActivation &source,
        const infinicore::Device &destination) override;

    PipelineTransportStats stats() const override;

private:
    mutable std::mutex stats_mutex_;
    size_t transfers_{0};
    size_t tensors_{0};
    size_t bytes_{0};
};

} // namespace infinilm::engine
