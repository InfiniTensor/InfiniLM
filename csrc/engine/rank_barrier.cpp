#include "rank_barrier.hpp"

namespace infinilm::engine {
RankBarrier::RankBarrier(size_t num_ranks) : thread_count_(num_ranks), generation_(0), arrived_(0) {}

bool RankBarrier::wait(bool success) {
    std::unique_lock<std::mutex> lock(mutex_);
    const size_t gen = generation_;
    generation_success_ = generation_success_ && success;

    if (++arrived_ == thread_count_) {
        completed_results_.push_back(generation_success_);
        generation_++;
        arrived_ = 0;
        generation_success_ = true;
        cv_.notify_all();
    } else {
        cv_.wait(lock, [&] { return gen != generation_; });
    }
    return completed_results_[gen];
}
} // namespace infinilm::engine
