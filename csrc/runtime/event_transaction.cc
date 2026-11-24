#include "event_transaction.hpp"
#include "runtime/owner_drain_fail_stop.hpp"
#include <infini/rt/device_guard.h>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::runtime {
namespace rt = infini::rt::runtime;
namespace {
void check(rt::Error status, const char *operation) {
    if (status != rt::kSuccess)
        throw std::runtime_error(std::string(operation)+" error="+std::to_string(static_cast<long long>(status)));
}
} // namespace

EventTransaction::EventTransaction(std::shared_ptr<Runtime> runtime,
                                   rt::Stream producer, rt::Stream consumer,
                                   unsigned int event_flags, Reuse reuse)
    : runtime_(std::move(runtime)), producer_(producer), consumer_(consumer),
      flags_(event_flags), reuse_(reuse) {
    if (!runtime_) throw std::invalid_argument("EventTransaction requires a Runtime");
    check_context();
    if (reuse_ != Reuse::PerTransaction && reuse_ != Reuse::Persistent)
        throw std::invalid_argument("invalid EventTransaction reuse policy");
    infini::rt::DeviceGuard guard{runtime_->device().type(), runtime_->device().index(), true};
    if (reuse_ == Reuse::Persistent) create_event();
}

EventTransaction::~EventTransaction() noexcept {
    try { close(); } catch (...) { std::terminate(); }
}

void EventTransaction::check_context() const {
    runtime_->assertOwnerThread();
    if (runtime_->isGraphRecording() || !runtime_->ownerDrainFailStopEnabled())
        throw std::logic_error("EventTransaction requires Eager owner drain fail-stop Runtime");
}

void EventTransaction::create_event() {
    rt::Event created = nullptr;
    check(rt::EventCreateWithFlags(&created, flags_), "EventCreateWithFlags");
    event_ = created;
}

void EventTransaction::destroy_event() {
    if (event_) {
        // A failed native destroy does not prove that the handle is reusable.
        const auto old = std::exchange(event_, nullptr);
        check(rt::EventDestroy(old), "EventDestroy");
    }
}

bool EventTransaction::submit() {
    check_context();
    if (state_ != State::Ready)
        throw std::logic_error("EventTransaction must complete before reuse; failed/closed transactions cannot retry");
    infini::rt::DeviceGuard guard{runtime_->device().type(), runtime_->device().index(), true};
    return submit_impl();
}

bool EventTransaction::submit_on_current_device() {
    check_context();
    if (state_ != State::Ready)
        throw std::logic_error("EventTransaction must complete before reuse; failed/closed transactions cannot retry");
    return submit_impl();
}

bool EventTransaction::submit_impl() {
    state_ = State::InFlight;
    try {
        if (reuse_ == Reuse::PerTransaction) create_event();
        check(rt::EventRecord(event_, producer_), "EventRecord");
        const auto status = rt::EventQuery(event_);
        const bool ready = status == rt::kSuccess;
        if (!ready && !rt::IsEventNotReady(status)) check(status, "EventQuery");
        check(rt::StreamWaitEvent(consumer_, event_, 0), "StreamWaitEvent");
        if (reuse_ == Reuse::PerTransaction) destroy_event();
        return ready;
    } catch (...) { state_ = State::Failed; throw; }
}

void EventTransaction::synchronize(rt::Stream stream) {
    try {
        const auto status = rt::StreamSynchronize(stream);
        detail::require_proven_owner_drain(true, status == rt::kSuccess);
        check(status, "StreamSynchronize");
    } catch (...) {
        detail::require_proven_owner_drain(true, false);
        throw;
    }
}

void EventTransaction::complete() {
    check_context();
    if (state_ == State::Ready) return;
    if (state_ != State::InFlight) throw std::logic_error("failed/closed EventTransaction cannot complete");
    infini::rt::DeviceGuard guard{runtime_->device().type(), runtime_->device().index(), true};
    synchronize(consumer_);
    state_ = State::Ready;
}

void EventTransaction::close() {
    if (state_ == State::Closed) return;
    check_context();
    infini::rt::DeviceGuard guard{runtime_->device().type(), runtime_->device().index(), true};
    if (state_ == State::Failed) {
        synchronize(producer_);
        if (consumer_ != producer_) synchronize(consumer_);
    } else if (state_ == State::InFlight) {
        synchronize(consumer_);
    }
    try {
        destroy_event();
        state_ = State::Closed;
    } catch (...) { state_ = State::Failed; throw; }
}

} // namespace infinilm::runtime
