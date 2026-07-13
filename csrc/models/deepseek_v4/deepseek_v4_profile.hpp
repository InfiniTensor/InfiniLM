#pragma once

#include "infinicore/context/context.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace infinilm::models::deepseek_v4::profile {

enum class Event : size_t {
    DecoderFfnLayer = 0,
    DecoderHcPre,
    DecoderFfnNorm,
    DecoderMoe,
    DecoderHcPost,
    MoeForward,
    MoeTopk,
    MoeExperts,
    MoeSharedExperts,
    MoeAllReduce,
    Count,
};

struct Stat {
    std::atomic<unsigned long long> calls{0};
    std::atomic<unsigned long long> micros{0};
};

inline std::array<Stat, static_cast<size_t>(Event::Count)> &stats() {
    static std::array<Stat, static_cast<size_t>(Event::Count)> value;
    return value;
}

inline const char *event_name(Event event) {
    switch (event) {
    case Event::DecoderFfnLayer: return "decoder.ffn_layer";
    case Event::DecoderHcPre: return "decoder.hc_pre";
    case Event::DecoderFfnNorm: return "decoder.ffn_norm";
    case Event::DecoderMoe: return "decoder.moe";
    case Event::DecoderHcPost: return "decoder.hc_post";
    case Event::MoeForward: return "moe.forward";
    case Event::MoeTopk: return "moe.topk";
    case Event::MoeExperts: return "moe.experts";
    case Event::MoeSharedExperts: return "moe.shared_experts";
    case Event::MoeAllReduce: return "moe.allreduce";
    case Event::Count: break;
    }
    return "unknown";
}

inline bool enabled() {
    static const bool value = [] {
        const char *env = std::getenv("INFINILM_DSV4_FFN_PROFILE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return value;
}

inline void dump() {
    if (!enabled()) {
        return;
    }
    std::fprintf(stderr, "\n[INFINILM_DSV4_FFN_PROFILE] aggregate GPU-synced wall time\n");
    for (size_t i = 0; i < static_cast<size_t>(Event::Count); ++i) {
        auto &stat = stats()[i];
        const auto calls = stat.calls.load(std::memory_order_relaxed);
        const auto micros = stat.micros.load(std::memory_order_relaxed);
        if (calls == 0) {
            continue;
        }
        const double total_ms = static_cast<double>(micros) / 1000.0;
        const double avg_ms = total_ms / static_cast<double>(calls);
        std::fprintf(stderr, "[INFINILM_DSV4_FFN_PROFILE] %-24s calls=%llu total_ms=%.3f avg_ms=%.6f\n",
                     event_name(static_cast<Event>(i)), calls, total_ms, avg_ms);
    }
}

inline void register_dump_once() {
    static std::once_flag flag;
    std::call_once(flag, [] { std::atexit(dump); });
}

class ScopedTimer {
public:
    explicit ScopedTimer(Event event)
        : event_(event), active_(enabled()) {
        if (active_) {
            register_dump_once();
            infinicore::context::syncStream();
            start_ = Clock::now();
        }
    }

    ~ScopedTimer() {
        if (!active_) {
            return;
        }
        infinicore::context::syncStream();
        const auto end = Clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        auto &stat = stats()[static_cast<size_t>(event_)];
        stat.calls.fetch_add(1, std::memory_order_relaxed);
        stat.micros.fetch_add(static_cast<unsigned long long>(us), std::memory_order_relaxed);
    }

private:
    using Clock = std::chrono::steady_clock;
    Event event_;
    bool active_{false};
    Clock::time_point start_{};
};

} // namespace infinilm::models::deepseek_v4::profile
