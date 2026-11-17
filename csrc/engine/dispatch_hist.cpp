#include "dispatch_hist.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <spdlog/spdlog.h>

namespace infinilm::engine::dispatch_hist {
namespace {

struct Counters {
    std::atomic<uint64_t> full{0};
    std::atomic<uint64_t> piecewise{0};
    std::atomic<uint64_t> none{0};
    std::atomic<uint64_t> none_eager_policy{0};
    std::atomic<uint64_t> none_mixed{0};
    std::atomic<uint64_t> none_multi_req_prefill{0};
    std::atomic<uint64_t> none_bucket_miss{0};
    std::atomic<uint64_t> none_decode_bs_miss{0};
    std::atomic<uint64_t> none_decode_bs_over_max{0};
    std::atomic<uint64_t> none_over_max{0};
    std::atomic<uint64_t> none_other{0};
    // Piecewise eager vs exact-CG (step-level; not per-layer).
    std::atomic<uint64_t> pw_exact_cg{0};
    std::atomic<uint64_t> pw_eager_pad_up{0};
    std::atomic<uint64_t> pw_eager_mid_chunk{0};
    std::atomic<uint64_t> pw_eager_missing_graph{0};
    std::atomic<uint64_t> steps_since_dump{0};
};

Counters &counters() {
    static Counters c;
    return c;
}

bool rank_worker_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_RANK_WORKER_PROFILE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

bool hang_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_HANG_TRACE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

bool cg_eager_hist_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_CG_EAGER_HIST");
        // Default off (quiet). LongBench / restart_serve set =1.
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

uint64_t dump_every_n() {
    static uint64_t cached = 0;
    if (cached == 0) {
        const char *raw = std::getenv("INFINI_CG_EAGER_HIST_EVERY");
        if (raw != nullptr && raw[0] != '\0') {
            const long v = std::strtol(raw, nullptr, 10);
            cached = v > 0 ? static_cast<uint64_t>(v) : 200;
        } else {
            cached = 200;
        }
    }
    return cached;
}

void dump_line(const char *tag) {
    const auto &h = counters();
    spdlog::info(
        "cg_eager_hist tag={} FULL={} PIECEWISE={} NONE={} "
        "pw_exact_cg={} pw_eager_pad_up={} pw_eager_mid_chunk={} pw_eager_missing={} "
        "none_over_max={} none_mixed={} none_bucket_miss={} none_decode_bs_miss={} "
        "none_decode_bs_over_max={} none_other={}",
        tag != nullptr ? tag : "-",
        h.full.load(std::memory_order_relaxed),
        h.piecewise.load(std::memory_order_relaxed),
        h.none.load(std::memory_order_relaxed),
        h.pw_exact_cg.load(std::memory_order_relaxed),
        h.pw_eager_pad_up.load(std::memory_order_relaxed),
        h.pw_eager_mid_chunk.load(std::memory_order_relaxed),
        h.pw_eager_missing_graph.load(std::memory_order_relaxed),
        h.none_over_max.load(std::memory_order_relaxed),
        h.none_mixed.load(std::memory_order_relaxed),
        h.none_bucket_miss.load(std::memory_order_relaxed),
        h.none_decode_bs_miss.load(std::memory_order_relaxed),
        h.none_decode_bs_over_max.load(std::memory_order_relaxed),
        h.none_other.load(std::memory_order_relaxed));
}

void atexit_dump() {
    if (cg_eager_hist_enabled()) {
        dump_line("atexit");
    }
}

} // namespace

void record_mode(CudaGraphRuntimeMode mode, const char *none_reason) {
    auto &h = counters();
    switch (mode) {
    case CudaGraphRuntimeMode::Full:
        h.full.fetch_add(1, std::memory_order_relaxed);
        break;
    case CudaGraphRuntimeMode::Piecewise:
        h.piecewise.fetch_add(1, std::memory_order_relaxed);
        break;
    case CudaGraphRuntimeMode::None:
    default:
        h.none.fetch_add(1, std::memory_order_relaxed);
        if (none_reason == nullptr) {
            h.none_other.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(none_reason, "eager_policy") == 0) {
            h.none_eager_policy.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(none_reason, "mixed") == 0) {
            h.none_mixed.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(none_reason, "multi_req_prefill") == 0) {
            h.none_multi_req_prefill.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(none_reason, "bucket_miss") == 0) {
            h.none_bucket_miss.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(none_reason, "decode_bs_miss") == 0) {
            h.none_decode_bs_miss.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(none_reason, "decode_bs_over_max") == 0) {
            h.none_decode_bs_over_max.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(none_reason, "over_max") == 0) {
            h.none_over_max.fetch_add(1, std::memory_order_relaxed);
        } else {
            h.none_other.fetch_add(1, std::memory_order_relaxed);
        }
        break;
    }
    maybe_dump_eager("periodic");
}

void record_piecewise_eager(bool pad_up, bool mid_chunk_eager, bool missing_graph) {
    auto &h = counters();
    if (pad_up) {
        h.pw_eager_pad_up.fetch_add(1, std::memory_order_relaxed);
    } else if (mid_chunk_eager) {
        h.pw_eager_mid_chunk.fetch_add(1, std::memory_order_relaxed);
    } else if (missing_graph) {
        h.pw_eager_missing_graph.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Final exact CG and exact-width mid CG (dual mid graphs) both land here.
        h.pw_exact_cg.fetch_add(1, std::memory_order_relaxed);
    }
}

void log_profile_hist(const char *tag) {
    if (!rank_worker_profile_enabled() && !hang_trace_enabled()) {
        return;
    }
    const auto &h = counters();
    spdlog::info(
        "{}: dispatch_hist FULL={} PIECEWISE={} NONE={} "
        "none_reason[eager_policy={} mixed={} multi_req_prefill={} "
        "bucket_miss={} decode_bs_miss={} decode_bs_over_max={} over_max={} other={}] "
        "pw[exact_cg={} pad_up={} mid_chunk={} missing={}]",
        tag,
        h.full.load(std::memory_order_relaxed),
        h.piecewise.load(std::memory_order_relaxed),
        h.none.load(std::memory_order_relaxed),
        h.none_eager_policy.load(std::memory_order_relaxed),
        h.none_mixed.load(std::memory_order_relaxed),
        h.none_multi_req_prefill.load(std::memory_order_relaxed),
        h.none_bucket_miss.load(std::memory_order_relaxed),
        h.none_decode_bs_miss.load(std::memory_order_relaxed),
        h.none_decode_bs_over_max.load(std::memory_order_relaxed),
        h.none_over_max.load(std::memory_order_relaxed),
        h.none_other.load(std::memory_order_relaxed),
        h.pw_exact_cg.load(std::memory_order_relaxed),
        h.pw_eager_pad_up.load(std::memory_order_relaxed),
        h.pw_eager_mid_chunk.load(std::memory_order_relaxed),
        h.pw_eager_missing_graph.load(std::memory_order_relaxed));
}

void maybe_dump_eager(const char *tag) {
    if (!cg_eager_hist_enabled()) {
        return;
    }
    ensure_atexit_dump();
    auto &h = counters();
    const uint64_t n = h.steps_since_dump.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n >= dump_every_n()) {
        h.steps_since_dump.store(0, std::memory_order_relaxed);
        dump_line(tag != nullptr ? tag : "periodic");
    }
}

void dump_eager_final(const char *tag) {
    if (!cg_eager_hist_enabled()) {
        return;
    }
    ensure_atexit_dump();
    dump_line(tag != nullptr ? tag : "teardown");
}

void ensure_atexit_dump() {
    static std::atomic<int> registered{0};
    int expected = 0;
    if (registered.compare_exchange_strong(expected, 1, std::memory_order_relaxed)) {
        std::atexit(atexit_dump);
    }
}

} // namespace infinilm::engine::dispatch_hist
