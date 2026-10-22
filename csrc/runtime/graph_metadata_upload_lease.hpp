#pragma once
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace infinilm::runtime {
// Dedicated serial-worker upload lease. No change to Tensor copy semantics.
// One stream drain retires the previous lease before its source is overwritten.
// Failed submission keeps the lease live; destruction drains or fails closed.
template <class RuntimeT, class MemoryT>
class GraphMetadataUploadLease {
public:
    struct Copy {
        void *destination;
        const void *source;
        size_t bytes;
    };
    explicit GraphMetadataUploadLease(RuntimeT &runtime) : runtime_(runtime) {}
    GraphMetadataUploadLease(const GraphMetadataUploadLease &) = delete;
    GraphMetadataUploadLease &operator=(const GraphMetadataUploadLease &) = delete;
    ~GraphMetadataUploadLease() noexcept {
        if (in_flight_) {
            try {
                runtime_.syncStream();
            } catch (...) { std::terminate(); }
        }
    }
    void drain() {
        runtime_.assertOwnerThread();
        if (in_flight_) {
            runtime_.syncStream();
            ++reuse_syncs;
            in_flight_ = false;
        }
    }
    void upload(const std::vector<Copy> &copies) {
        runtime_.assertOwnerThread();
        size_t total = 0;
        for (const auto &copy : copies) {
            if (!copy.source || !copy.destination || copy.bytes > std::numeric_limits<size_t>::max() - total) {
                throw std::invalid_argument("invalid graph metadata upload");
            }
            total += copy.bytes;
        }
        drain();
        if (!host_ || host_->size() < total) {
            auto host = runtime_.allocatePinnedHostMemory(total);
            auto device = runtime_.allocateExactMemory(total);
            host_ = std::move(host);
            device_ = std::move(device);
        }
        size_t offset = 0;
        for (const auto &copy : copies) {
            std::memcpy(host_->data() + offset, copy.source, copy.bytes);
            offset += copy.bytes;
        }
        // Set before the first provider call: even a throwing call can enqueue.
        in_flight_ = true;
        runtime_.memcpyH2D(device_->data(), host_->data(), total, true);
        ++uploads;
        offset = 0;
        for (const auto &copy : copies) {
            runtime_.memcpyD2D(copy.destination, device_->data() + offset, copy.bytes, true);
            offset += copy.bytes;
        }
    }
    size_t uploads = 0;
    size_t reuse_syncs = 0;

private:
    RuntimeT &runtime_;
    std::shared_ptr<MemoryT> host_;
    std::shared_ptr<MemoryT> device_;
    bool in_flight_ = false;
};
} // namespace infinilm::runtime
