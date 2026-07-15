#pragma once

#include <infinicore/device.hpp>
#include <infinicore/tensor.hpp>

#include <cstddef>
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

class PipelineTransport {
public:
    virtual ~PipelineTransport() = default;

    // The returned tensors are ready for use on destination and no longer
    // depend on the lifetime of the source activation.
    virtual PipelineActivation transfer(
        const PipelineActivation &source,
        const infinicore::Device &destination) = 0;

    virtual PipelineTransportStats stats() const = 0;
};

class PeerCopyTransport final : public PipelineTransport {
public:
    PipelineActivation transfer(
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
