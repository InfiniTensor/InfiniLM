#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "runtime/tensor.hpp"

namespace infinilm::runtime {
class Runtime;
}

namespace infinilm::execution::graph {
// Forward declarations
class GraphManager;

class GraphTensor : public ::infinilm::runtime::Tensor {
public:
    enum class SnapshotPolicy {
        kRecordingAware,
        kBlob,
    };

    GraphTensor(::infinilm::runtime::Runtime &runtime, const ::infinilm::runtime::Tensor &);
    GraphTensor(::infinilm::runtime::Runtime &runtime, const ::infinilm::runtime::Tensor &, SnapshotPolicy policy);
};

class RecordedCall final {
public:
    explicit RecordedCall(std::function<void()> call,
                          bool device_graph_capture_safe,
                          bool explicit_host_replay = true,
                          std::string name = "RecordedCall",
                          bool host_requires_synchronization = true)
        : call_(std::move(call)),
          device_graph_capture_safe_(device_graph_capture_safe),
          explicit_host_replay_(explicit_host_replay), name_(std::move(name)),
          host_requires_synchronization_(host_requires_synchronization) {}

    void run() const { call_(); }
    const std::string &name() const { return name_; }
    bool host_requires_synchronization() const {
        return host_requires_synchronization_;
    }
    bool is_device_graph_capture_safe() const {
        return device_graph_capture_safe_;
    }
    bool is_explicit_host_replay() const {
        return !device_graph_capture_safe_ && explicit_host_replay_;
    }

private:
    std::function<void()> call_;
    bool device_graph_capture_safe_;
    bool explicit_host_replay_;
    std::string name_;
    bool host_requires_synchronization_;
};

class Graph {
private:
    // Declared first so it outlives operators and all segment device graphs.
    std::shared_ptr<::infinilm::runtime::Runtime> runtime_lease_;
    std::shared_ptr<void> allocation_lease_;

public:
    struct ExecutionReport {
        size_t operator_count = 0;
        size_t device_graph_segments = 0;
        size_t host_segments = 0;
        bool capture_attempted = false;
        std::string fallback_reason;
    };

    Graph();
    ~Graph() noexcept;

    void run() const;
    ExecutionReport execution_report() const;

protected:
    void add_operator(std::shared_ptr<RecordedCall> op);
    void instantiate();
    std::vector<std::shared_ptr<RecordedCall>> op_list_;

    friend class GraphManager;

private:
    bool require_device_graph_ = false;
    mutable bool replay_failed_ = false; // Partial replay cannot be safely retried.
    std::thread::id owner_thread_;
    ExecutionReport execution_report_;
    void retain_runtime(std::shared_ptr<::infinilm::runtime::Runtime> runtime,
                        std::shared_ptr<void> allocation_lease);

    struct DeviceGraph;
    struct Segment {
        explicit Segment(bool capture_safe);
        ~Segment() noexcept;

        void run() const;

        bool capture_safe;
        std::vector<std::shared_ptr<RecordedCall>> ops;
        std::unique_ptr<DeviceGraph> device_graph_;
    };
    std::vector<std::unique_ptr<Segment>> segments_;

    friend class ::infinilm::runtime::Runtime;
    friend class GraphManager;
};
} // namespace infinilm::execution::graph

#ifdef ENABLE_MUTUAL_AWARENESS
#include "execution/trace/op_trace.hpp"
#include "execution/trace/op_type_registry.hpp"

// Trace one op invocation into the global ring. Op type is resolved by
// stringified class name through `opTypeFromName`, so new graph ops are
// automatically discoverable without modifying the op header.
#define INFINILM_TRACE_OP(__OP_NAME__, __TRACE_TENSOR__)                             \
    do {                                                                             \
        auto __op_type = ::infinilm::execution::trace::opTypeFromName(#__OP_NAME__); \
        auto &&__trace_tensor = (__TRACE_TENSOR__);                                  \
        if (__trace_tensor) {                                                        \
            const auto &__trace_shape = __trace_tensor->shape();                     \
            const auto __trace_device = __trace_tensor->device();                    \
            ::infinilm::execution::trace::traceOp(                                   \
                __op_type,                                                           \
                __trace_shape.data(),                                                \
                __trace_shape.size(),                                                \
                static_cast<uint8_t>(__trace_tensor->dtype()),                       \
                static_cast<uint8_t>(__trace_device.type()),                         \
                static_cast<int8_t>(__trace_device.index()));                        \
        } else {                                                                     \
            ::infinilm::execution::trace::traceOp(__op_type, nullptr, 0, 0, 0, -1);  \
        }                                                                            \
    } while (0)
#else
#define INFINILM_TRACE_OP(__OP_NAME__, __TRACE_TENSOR__) ((void)0)
#endif
