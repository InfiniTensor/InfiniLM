#pragma once

#include "runtime/device.hpp"

#include <infini/rt.h>

#include <memory>

namespace infinilm::runtime {

class Runtime;

// Cross-runtime completion fence used to order device work without a host sync.
class StreamEvent : public std::enable_shared_from_this<StreamEvent> {
public:
    static std::shared_ptr<StreamEvent> create(Runtime &runtime);
    ~StreamEvent() noexcept;

    StreamEvent(const StreamEvent &) = delete;
    StreamEvent &operator=(const StreamEvent &) = delete;

    void record(Runtime &runtime);
    void wait(Runtime &runtime) const;
    Device device() const { return device_; }

private:
    explicit StreamEvent(Runtime &runtime);

    Device device_;
    infini::rt::runtime::Event event_{nullptr};
};

} // namespace infinilm::runtime
