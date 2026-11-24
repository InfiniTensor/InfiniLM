#pragma once

#include "runtime/runtime.hpp"
#include <memory>

namespace infinilm::runtime {

// Eager record/query/one-wait transaction. Both streams are borrowed and must
// belong to this Runtime's device and outlive the transaction. Producer work is
// already submitted at entry. The native backend's flags are supplied explicitly
// (HYGON qualification uses cudaEventDisableTiming, never an old ABI literal).
// All operations and destruction run on the Runtime's owning thread.
class EventTransaction {
public:
    enum class Reuse { PerTransaction, Persistent };
    enum class State { Ready, InFlight, Failed, Closed };
    EventTransaction(std::shared_ptr<Runtime> runtime,
                     infini::rt::runtime::Stream producer,
                     infini::rt::runtime::Stream consumer,
                     unsigned int event_flags, Reuse reuse);
    ~EventTransaction() noexcept;
    EventTransaction(const EventTransaction &) = delete;
    EventTransaction &operator=(const EventTransaction &) = delete;

    // Returns the query's ready state. It always submits exactly one wait on
    // success, including when the query reports ready. No polling or host sync.
    bool submit();
    // Specialized fixed-context entry. The caller must select this Runtime's
    // device type AND ordinal before entry and keep them unchanged during the
    // call. The owner and borrowed-stream lifetime requirements still apply.
    // No device query, selection or restoration is performed here. Call submit()
    // when the current device may differ; complete()/close() remain guarded.
    bool submit_on_current_device();
    // Outside the API timing window: proves consumer completion before reuse.
    void complete();
    void close();
    State state() const noexcept { return state_; }

private:
    void check_context() const;
    bool submit_impl();
    void create_event();
    void destroy_event();
    void synchronize(infini::rt::runtime::Stream stream);

    std::shared_ptr<Runtime> runtime_;
    infini::rt::runtime::Stream producer_, consumer_;
    unsigned int flags_;
    Reuse reuse_;
    infini::rt::runtime::Event event_ = nullptr;
    State state_ = State::Ready;
};

} // namespace infinilm::runtime
