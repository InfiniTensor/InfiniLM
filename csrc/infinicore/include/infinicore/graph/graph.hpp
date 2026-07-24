#pragma once

#include <memory>
#include <vector>

#include "../tensor.hpp"

namespace infinicore {
class Runtime;
}

namespace infinicore::graph {
// Forward declarations
class GraphManager;

class GraphTensor : public Tensor {
public:
    GraphTensor(const Tensor &);
};

class GraphOperator {
public:
    virtual void run() const = 0;
    virtual ~GraphOperator() = default;
};

class DispatchableGraphOperator : public GraphOperator {
public:
    void run() const override;
    ~DispatchableGraphOperator() override;

protected:
    using run_schema = void (*)(void *);
    using cleanup_schema = void (*)(void **);
    void *planned_meta_;
    run_schema runner_;
    cleanup_schema deleter_;
};

class Graph {
private:
    // Declared first so it outlives operators and the native device graph.
    std::shared_ptr<::infinicore::Runtime> runtime_lease_;
    std::shared_ptr<void> allocation_lease_;

public:
    Graph();
    ~Graph() noexcept;

    void run() const;

protected:
    void add_operator(std::shared_ptr<GraphOperator> op);
    void instantiate();
    std::vector<std::shared_ptr<GraphOperator>> op_list_;

    friend class GraphManager;

private:
    void retain_runtime(std::shared_ptr<::infinicore::Runtime> runtime,
                        std::shared_ptr<void> allocation_lease);

    struct DeviceGraph;
    std::unique_ptr<DeviceGraph> device_graph_;

    friend class ::infinicore::Runtime;
};
} // namespace infinicore::graph

#define INFINICORE_GRAPH_OP_CLASS(__OP_NAME__, ...)                        \
    class __OP_NAME__ : public graph::DispatchableGraphOperator {          \
    public:                                                                \
        using schema = void (*)(__VA_ARGS__);                              \
        using plan_schema = void *(*)(__VA_ARGS__);                        \
        static common::OpDispatcher<plan_schema> &plan_dispatcher();       \
        static common::OpDispatcher<run_schema> &run_dispatcher();         \
        static common::OpDispatcher<cleanup_schema> &cleanup_dispatcher(); \
        __OP_NAME__(__VA_ARGS__);                                          \
        static void execute(__VA_ARGS__);                                  \
    };

#define INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(__OP_NAME__)                                  \
    common::OpDispatcher<__OP_NAME__::plan_schema> &__OP_NAME__::plan_dispatcher() {       \
        static common::OpDispatcher<__OP_NAME__::plan_schema> dispatcher_;                 \
        return dispatcher_;                                                                \
    }                                                                                      \
    common::OpDispatcher<__OP_NAME__::run_schema> &__OP_NAME__::run_dispatcher() {         \
        static common::OpDispatcher<__OP_NAME__::run_schema> dispatcher_;                  \
        return dispatcher_;                                                                \
    }                                                                                      \
    common::OpDispatcher<__OP_NAME__::cleanup_schema> &__OP_NAME__::cleanup_dispatcher() { \
        static common::OpDispatcher<__OP_NAME__::cleanup_schema> dispatcher_;              \
        return dispatcher_;                                                                \
    }

#define INFINICORE_GRAPH_OP_DISPATCH(__DEVICE_TYPE__, ...)                  \
    planned_meta_ = plan_dispatcher().lookup(__DEVICE_TYPE__)(__VA_ARGS__); \
    runner_ = run_dispatcher().lookup(__DEVICE_TYPE__);                     \
    deleter_ = cleanup_dispatcher().lookup(__DEVICE_TYPE__);

#define INFINICORE_DETAIL_FIRST_ARG(__FIRST__, ...) __FIRST__

#ifdef ENABLE_MUTUAL_AWARENESS
#include "../analyzer/op_trace.hpp"
#include "../analyzer/op_type_registry.hpp"

// Trace one op invocation into the global ring. Op type is resolved by
// stringified class name through `opTypeFromName`, so new graph ops are
// automatically discoverable without modifying the op header.
#define _INFINICORE_TRACE_OP(__OP_NAME__, __TRACE_TENSOR__)                    \
    do {                                                                       \
        auto __op_type = ::infinicore::analyzer::opTypeFromName(#__OP_NAME__); \
        auto &&__trace_tensor = (__TRACE_TENSOR__);                            \
        if (__trace_tensor) {                                                  \
            const auto &__trace_shape = __trace_tensor->shape();               \
            const auto __trace_device = __trace_tensor->device();              \
            ::infinicore::analyzer::traceOp(                                   \
                __op_type,                                                     \
                __trace_shape.data(),                                          \
                __trace_shape.size(),                                          \
                static_cast<uint8_t>(__trace_tensor->dtype()),                 \
                static_cast<uint8_t>(__trace_device.type()),                   \
                static_cast<int8_t>(__trace_device.index()));                  \
        } else {                                                               \
            ::infinicore::analyzer::traceOp(__op_type, nullptr, 0, 0, 0, -1);  \
        }                                                                      \
    } while (0)
#else
#define _INFINICORE_TRACE_OP(__OP_NAME__, __TRACE_TENSOR__) ((void)0)
#endif

#define INFINICORE_GRAPH_OP_RECORD_OR_RUN(__OP_NAME__, ...)  \
    const bool ___recording = context::isGraphRecording();   \
    auto ___op = std::make_shared<__OP_NAME__>(__VA_ARGS__); \
    if (___recording) {                                      \
        context::addGraphOperator(___op);                    \
    } else {                                                 \
        ___op->run();                                        \
    }                                                        \
    _INFINICORE_TRACE_OP(__OP_NAME__, INFINICORE_DETAIL_FIRST_ARG(__VA_ARGS__));

#define INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(__OP_NAME__, __PLAN_F__, __RUN_F__, __CLEANUP_F__) \
    static bool registered = []() {                                                               \
        __OP_NAME__::plan_dispatcher().registerAll(__PLAN_F__, false);                            \
        __OP_NAME__::run_dispatcher().registerAll(__RUN_F__, false);                              \
        __OP_NAME__::cleanup_dispatcher().registerAll(__CLEANUP_F__, false);                      \
        return true;                                                                              \
    }();
