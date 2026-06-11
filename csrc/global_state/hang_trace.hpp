#pragma once

#include <chrono>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include <string>

namespace infinilm::global_state::hang_trace {

inline bool enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_HANG_TRACE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

inline double slow_ms() {
    static double cached = -1.0;
    if (cached < 0.0) {
        const char *raw = std::getenv("INFINI_HANG_TRACE_SLOW_MS");
        if (raw != nullptr && raw[0] != '\0') {
            cached = std::atof(raw);
        }
        if (cached <= 0.0) {
            cached = 5000.0;
        }
    }
    return cached;
}

inline double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

class ScopedBracket {
public:
    ScopedBracket(const char *tag, int tp_rank = -1)
        : tag_(tag), tp_rank_(tp_rank), t0_(monotonic_ms()) {
        if (enabled()) {
            if (tp_rank_ >= 0) {
                spdlog::info("hang_trace: {} enter tp_rank={}", tag_, tp_rank_);
            } else {
                spdlog::info("hang_trace: {} enter", tag_);
            }
        }
    }

    ~ScopedBracket() {
        if (!enabled()) {
            return;
        }
        const double ms = monotonic_ms() - t0_;
        if (tp_rank_ >= 0) {
            spdlog::info("hang_trace: {} exit tp_rank={} ms={:.3f}", tag_, tp_rank_, ms);
        } else {
            spdlog::info("hang_trace: {} exit ms={:.3f}", tag_, ms);
        }
        if (ms >= slow_ms()) {
            if (tp_rank_ >= 0) {
                spdlog::warn(
                    "hang_trace: STALL tag={} tp_rank={} ms={:.3f} slow_ms={:.0f}",
                    tag_,
                    tp_rank_,
                    ms,
                    slow_ms());
            } else {
                spdlog::warn(
                    "hang_trace: STALL tag={} ms={:.3f} slow_ms={:.0f}",
                    tag_,
                    ms,
                    slow_ms());
            }
        }
    }

    ScopedBracket(const ScopedBracket &) = delete;
    ScopedBracket &operator=(const ScopedBracket &) = delete;

private:
    const char *tag_;
    int tp_rank_;
    double t0_;
};

} // namespace infinilm::global_state::hang_trace
