#include "infinicore/analyzer/op_trace.hpp"

namespace infinicore::analyzer {

OpTraceRing &getGlobalOpTrace() {
    static OpTraceRing instance(OpTraceRing::DEFAULT_CAPACITY);
    return instance;
}

} // namespace infinicore::analyzer
