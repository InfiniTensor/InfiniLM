#pragma once

#include <infiniccl.h>
#include <infinicore/device.hpp>
#include <infinicore/tensor.hpp>
#include <infinirt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace infinicore {
class DeviceEvent;
}

namespace infinilm::engine::distributed {

struct PendingCollective {
    std::size_t slot = 0;
    std::uint64_t generation = 0;
    bool valid = false;
};

class AsyncCollectiveContext {
public:
    explicit AsyncCollectiveContext(const infinicore::Device &device, bool enabled);
    ~AsyncCollectiveContext();

    AsyncCollectiveContext(const AsyncCollectiveContext &) = delete;
    AsyncCollectiveContext &operator=(const AsyncCollectiveContext &) = delete;

    bool enabled() const { return enabled_; }
    infinirtStream_t stream() const { return stream_; }

    PendingCollective allreduce(infinicore::Tensor output,
                                const infinicore::Tensor &input,
                                infinicclReduceOp_t op,
                                infinicclComm_t communicator);
    void wait(const PendingCollective &pending);
    void allreduce_and_wait(infinicore::Tensor output,
                            const infinicore::Tensor &input,
                            infinicclReduceOp_t op,
                            infinicclComm_t communicator);

private:
    struct EventSlot;

    EventSlot &next_event_slot();
    void validate_allreduce(infinicore::Tensor output,
                            const infinicore::Tensor &input,
                            infinicclComm_t communicator) const;

    infinicore::Device device_;
    bool enabled_ = false;
    infinirtStream_t stream_ = nullptr;
    std::vector<std::unique_ptr<EventSlot>> event_slots_;
    std::size_t next_slot_ = 0;
    std::uint64_t generation_ = 1;
};

class AsyncCollectiveContextGuard {
public:
    explicit AsyncCollectiveContextGuard(AsyncCollectiveContext *context);
    ~AsyncCollectiveContextGuard();

    AsyncCollectiveContextGuard(const AsyncCollectiveContextGuard &) = delete;
    AsyncCollectiveContextGuard &operator=(const AsyncCollectiveContextGuard &) = delete;

private:
    AsyncCollectiveContext *previous_ = nullptr;
};

bool async_collectives_enabled_by_env();
bool async_collectives_force_enabled_by_env();
bool async_collectives_supported_by_policy(const infinicore::Device &device, int tp_size);
bool async_collectives_enabled_for_rank(const infinicore::Device &device, int tp_size, infinicclComm_t communicator);
AsyncCollectiveContext *maybe_current_async_collective_context();
AsyncCollectiveContext &current_async_collective_context();

} // namespace infinilm::engine::distributed
