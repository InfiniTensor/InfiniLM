#include "execution/trace/op_trace.hpp"

namespace infinilm::execution::trace {

OpTraceRing &getGlobalOpTrace() {
    static OpTraceRing instance(OpTraceRing::DEFAULT_CAPACITY);
    return instance;
}

} // namespace infinilm::execution::trace
