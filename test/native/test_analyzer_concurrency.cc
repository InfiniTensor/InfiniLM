#include "infinicore/analyzer/phase_detector.hpp"

#include <atomic>
#include <cassert>
#include <stdexcept>
#include <thread>

using infinicore::analyzer::OpTraceEntry;
using infinicore::analyzer::OpTraceRing;
using infinicore::analyzer::OpType;
using infinicore::analyzer::PhaseDetector;
using infinicore::analyzer::PhaseType;

int main() {
    try {
        OpTraceRing invalid(0);
        assert(false);
    } catch (const std::invalid_argument &) {
    }

    OpTraceRing trace(8);
    std::atomic<bool> done{false};
    std::thread writer([&]() {
        for (uint32_t value = 1; value <= 100000; ++value) {
            OpTraceEntry entry;
            entry.op_type = OpType::MHA_KVCACHE;
            entry.ndim = OpTraceEntry::MAX_DIMS;
            for (auto &dim : entry.shape) {
                dim = value;
            }
            entry.timestamp_ns = value;
            trace.write(entry);
            if (value % 997 == 0) {
                trace.clear();
            }
        }
        done.store(true, std::memory_order_release);
    });
    size_t snapshots = 0;
    do {
        const auto entries = trace.getAllEntries();
        uint64_t previous = 0;
        for (const auto &entry : entries) {
            assert(entry.timestamp_ns > previous);
            previous = entry.timestamp_ns;
            assert(entry.ndim == OpTraceEntry::MAX_DIMS);
            for (const auto dim : entry.shape) {
                assert(dim == entry.timestamp_ns);
            }
        }
        ++snapshots;
    } while (!done.load(std::memory_order_acquire));
    writer.join();
    assert(snapshots > 0);
    assert(trace.size() == trace.capacity());
    assert(trace.getRecentEntries(3).size() == 3);
    assert(trace.getRecentEntries(0).empty());

    PhaseDetector detector;
    OpTraceEntry attention;
    attention.op_type = OpType::MHA_KVCACHE;
    const size_t decode_shape[] = {1, 1, 32, 128};
    attention.setShape(decode_shape, 4);
    assert(detector.detect({attention}) == PhaseType::DECODE);
    attention.shape[1] = 64;
    attention.shape[2] = 1;
    assert(detector.detect({attention}) == PhaseType::PREFILL);

    attention.op_type = OpType::CAUSAL_SOFTMAX;
    const size_t score_shape[] = {32, 1, 4096};
    attention.setShape(score_shape, 3);
    assert(detector.detect({attention}) == PhaseType::DECODE);

    attention.op_type = OpType::MHA_VARLEN;
    const size_t packed_shape[] = {64, 32, 128};
    attention.setShape(packed_shape, 3);
    assert(detector.detect({attention}) == PhaseType::ATTENTION_DENSE);
    attention.op_type = OpType::MHA_KVCACHE;
    assert(detector.detect({attention}) == PhaseType::ATTENTION_DENSE);
}
