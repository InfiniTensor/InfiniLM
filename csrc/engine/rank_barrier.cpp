#include "rank_barrier.hpp"

namespace infinilm::engine {
RankBarrier::RankBarrier(size_t num_ranks) : thread_count_(num_ranks), generation_(0), arrived_(0) {}

void RankBarrier::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;

    if (++arrived_ == thread_count_) {
        // last thread
        generation_++;
        arrived_ = 0;
        cv_.notify_all();
    } else {
        cv_.wait(lock, [&] { return gen != generation_; });
    }
}
} // namespace infinilm::engine
