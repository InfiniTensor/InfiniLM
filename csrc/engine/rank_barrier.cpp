#include "rank_barrier.hpp"

#include "../global_state/ar_profile.hpp"
#include "../global_state/hang_trace.hpp"

#include <spdlog/spdlog.h>
#include <string>

namespace infinilm::engine {
RankBarrier::RankBarrier(size_t num_ranks) : thread_count_(num_ranks), generation_(0), arrived_(0) {}

void RankBarrier::wait(const char *label, int tp_rank) {
    const bool trace = global_state::hang_trace::enabled();
    const bool ar_profile = global_state::ar_profile::enabled();
    const bool timed = trace || ar_profile;
    const double t0 = timed ? global_state::ar_profile::monotonic_ms() : 0.0;
    size_t arrived_before = 0;
    int gen_before = 0;


    {
        std::unique_lock<std::mutex> lock(mutex_);
        gen_before = static_cast<int>(generation_);
        arrived_before = arrived_;
        if (trace) {
            if (label != nullptr && label[0] != '\0') {
                if (tp_rank >= 0) {
                    spdlog::info(
                        "hang_trace: barrier_wait_enter label={} tp_rank={} gen={} arrived={}/{}",
                        label,
                        tp_rank,
                        gen_before,
                        arrived_before,
                        thread_count_);
                } else {
                    spdlog::info(
                        "hang_trace: barrier_wait_enter label={} gen={} arrived={}/{}",
                        label,
                        gen_before,
                        arrived_before,
                        thread_count_);
                }
            } else {
                spdlog::info(
                    "hang_trace: barrier_wait_enter gen={} arrived={}/{}",
                    gen_before,
                    arrived_before,
                    thread_count_);
            }
        }

        int gen = generation_;

        if (++arrived_ == thread_count_) {
            generation_++;
            arrived_ = 0;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [&] { return gen != generation_; });
        }
    }


    if (timed) {
        const double ms = global_state::ar_profile::monotonic_ms() - t0;
        if (ar_profile) {
            global_state::ar_profile::record_barrier_wait(label, ms);
        }
        if (trace) {
        if (label != nullptr && label[0] != '\0') {
            if (tp_rank >= 0) {
                spdlog::info(
                    "hang_trace: barrier_wait_exit label={} tp_rank={} ms={:.3f}",
                    label,
                    tp_rank,
                    ms);
            } else {
                spdlog::info(
                    "hang_trace: barrier_wait_exit label={} ms={:.3f}",
                    label,
                    ms);
            }
        } else {
            spdlog::info("hang_trace: barrier_wait_exit ms={:.3f}", ms);
        }
        if (ms >= global_state::hang_trace::slow_ms()) {
            if (label != nullptr && label[0] != '\0' && tp_rank >= 0) {
                spdlog::warn(
                    "hang_trace: STALL tag=barrier_wait label={} tp_rank={} ms={:.3f}",
                    label,
                    tp_rank,
                    ms);
            } else if (label != nullptr && label[0] != '\0') {
                spdlog::warn(
                    "hang_trace: STALL tag=barrier_wait label={} ms={:.3f}",
                    label,
                    ms);
            } else {
                spdlog::warn("hang_trace: STALL tag=barrier_wait ms={:.3f}", ms);
            }
        }
        }
    }
}
} // namespace infinilm::engine
