#include "async_collective.hpp"

#include <infinicore/context/context.hpp>
#include <infinicore/device_event.hpp>

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace infinilm::engine::distributed {
namespace {

thread_local AsyncCollectiveContext *current_context = nullptr;

std::string trim(const std::string &value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

void check_status(infiniStatus_t status, const char *what) {
    if (status != INFINI_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(static_cast<int>(status)));
    }
}

bool env_flag_enabled(const char *name) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return false;
    }
    const std::string value = to_lower(trim(raw));
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

} // namespace

struct AsyncCollectiveContext::EventSlot {
    EventSlot(const infinicore::Device &device, std::uint64_t initial_generation)
        : compute_ready(device, INFINIRT_EVENT_DISABLE_TIMING),
          collective_done(device, INFINIRT_EVENT_DISABLE_TIMING),
          generation(initial_generation) {}

    infinicore::DeviceEvent compute_ready;
    infinicore::DeviceEvent collective_done;
    std::uint64_t generation = 0;
};

bool async_collectives_enabled_by_env() {
    return env_flag_enabled("INFINILM_ENABLE_ASYNC_COLLECTIVES") ||
           env_flag_enabled("INFINILM_ENABLE_COMM_OVERLAP");
}

bool async_collectives_force_enabled_by_env() {
    return env_flag_enabled("INFINILM_COMM_OVERLAP_FORCE");
}

bool async_collectives_supported_by_policy(const infinicore::Device &device, int tp_size) {
    if (device.getType() != infinicore::Device::Type::NVIDIA || tp_size <= 1) {
        return false;
    }
    return tp_size == 2 || async_collectives_force_enabled_by_env();
}

bool async_collectives_enabled_for_rank(const infinicore::Device &device, int tp_size, infinicclComm_t communicator) {
    return communicator != nullptr &&
           async_collectives_enabled_by_env() &&
           async_collectives_supported_by_policy(device, tp_size);
}

AsyncCollectiveContext::AsyncCollectiveContext(const infinicore::Device &device, bool enabled)
    : device_(device), enabled_(enabled) {
    if (!enabled_) {
        return;
    }

    auto previous_device = infinicore::context::getDevice();
    if (previous_device != device_) {
        infinicore::context::setDevice(device_);
    }

    try {
        check_status(infinirtStreamCreate(&stream_), "infinirtStreamCreate");
        const std::size_t event_pool_size = 64;
        event_slots_.reserve(event_pool_size);
        for (std::size_t i = 0; i < event_pool_size; ++i) {
            event_slots_.push_back(std::make_unique<EventSlot>(device_, generation_++));
        }
    } catch (...) {
        event_slots_.clear();
        if (stream_ != nullptr) {
            (void)infinirtStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (previous_device != device_) {
            infinicore::context::setDevice(previous_device);
        }
        throw;
    }

    if (previous_device != device_) {
        infinicore::context::setDevice(previous_device);
    }
}

AsyncCollectiveContext::~AsyncCollectiveContext() {
    if (stream_ == nullptr) {
        return;
    }

    try {
        auto previous_device = infinicore::context::getDevice();
        if (previous_device != device_) {
            infinicore::context::setDevice(device_);
        }
        (void)infinirtStreamSynchronize(stream_);
        event_slots_.clear();
        (void)infinirtStreamDestroy(stream_);
        stream_ = nullptr;
        if (previous_device != device_) {
            infinicore::context::setDevice(previous_device);
        }
    } catch (...) {
        stream_ = nullptr;
    }
}

AsyncCollectiveContext::EventSlot &AsyncCollectiveContext::next_event_slot() {
    if (event_slots_.empty()) {
        throw std::runtime_error("async collective event pool is not initialized");
    }
    EventSlot &slot = *event_slots_[next_slot_];
    next_slot_ = (next_slot_ + 1) % event_slots_.size();
    slot.generation = generation_++;
    return slot;
}

void AsyncCollectiveContext::validate_allreduce(infinicore::Tensor output,
                                                const infinicore::Tensor &input,
                                                infinicclComm_t communicator) const {
    if (!enabled_ || stream_ == nullptr) {
        throw std::runtime_error("async collective context is disabled");
    }
    if (communicator == nullptr) {
        throw std::runtime_error("async allreduce requires a non-null communicator");
    }
    if (!output || !input) {
        throw std::runtime_error("async allreduce received an empty tensor");
    }
    if (output->device() != input->device() || output->device() != device_) {
        throw std::runtime_error("async allreduce tensors must be on the context device");
    }
    if (!output->is_contiguous() || !input->is_contiguous()) {
        throw std::runtime_error("async allreduce requires contiguous tensors");
    }
    if (output->numel() != input->numel()) {
        throw std::runtime_error("async allreduce input/output numel mismatch");
    }
    if (output->dtype() != input->dtype()) {
        throw std::runtime_error("async allreduce input/output dtype mismatch");
    }
}

PendingCollective AsyncCollectiveContext::allreduce(infinicore::Tensor output,
                                                    const infinicore::Tensor &input,
                                                    infinicclReduceOp_t op,
                                                    infinicclComm_t communicator) {
    validate_allreduce(output, input, communicator);

    auto previous_device = infinicore::context::getDevice();
    if (previous_device != device_) {
        infinicore::context::setDevice(device_);
    }

    EventSlot &slot = next_event_slot();
    slot.compute_ready.record(infinicore::context::getStream());
    slot.compute_ready.wait(stream_);

    check_status(infinicclAllReduce(
                     const_cast<std::byte *>(input->data()),
                     output->data(),
                     input->numel(),
                     static_cast<infiniDtype_t>(static_cast<int>(input->dtype())),
                     op,
                     communicator,
                     stream_),
                 "infinicclAllReduce");

    slot.collective_done.record(stream_);

    if (previous_device != device_) {
        infinicore::context::setDevice(previous_device);
    }

    return PendingCollective{(next_slot_ + event_slots_.size() - 1) % event_slots_.size(), slot.generation, true};
}

void AsyncCollectiveContext::wait(const PendingCollective &pending) {
    if (!pending.valid) {
        return;
    }
    if (pending.slot >= event_slots_.size()) {
        throw std::runtime_error("invalid pending collective slot");
    }
    EventSlot &slot = *event_slots_[pending.slot];
    if (slot.generation != pending.generation) {
        throw std::runtime_error("pending collective event slot was reused before wait");
    }

    auto previous_device = infinicore::context::getDevice();
    if (previous_device != device_) {
        infinicore::context::setDevice(device_);
    }
    slot.collective_done.wait(infinicore::context::getStream());
    if (previous_device != device_) {
        infinicore::context::setDevice(previous_device);
    }
}

void AsyncCollectiveContext::allreduce_and_wait(infinicore::Tensor output,
                                                const infinicore::Tensor &input,
                                                infinicclReduceOp_t op,
                                                infinicclComm_t communicator) {
    auto pending = allreduce(output, input, op, communicator);
    wait(pending);
}

AsyncCollectiveContextGuard::AsyncCollectiveContextGuard(AsyncCollectiveContext *context)
    : previous_(current_context) {
    current_context = context;
}

AsyncCollectiveContextGuard::~AsyncCollectiveContextGuard() {
    current_context = previous_;
}

AsyncCollectiveContext *maybe_current_async_collective_context() {
    return current_context;
}

AsyncCollectiveContext &current_async_collective_context() {
    if (current_context == nullptr) {
        throw std::runtime_error("no active async collective context");
    }
    return *current_context;
}

} // namespace infinilm::engine::distributed
