#include "rank_barrier.hpp"

#include <stdexcept>

namespace infinilm::engine {
RankBarrier::RankBarrier(size_t num_ranks) : thread_count_(num_ranks) {
    if (num_ranks == 0) {
        throw std::invalid_argument("RankBarrier requires at least one rank");
    }
}

void RankBarrier::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (broken_) {
        auto error = first_error_;
        lock.unlock();
        if (error) {
            std::rethrow_exception(error);
        }
        throw std::runtime_error("rank barrier aborted");
    }

    const size_t generation = generation_;

    if (++arrived_ == thread_count_) {
        // last thread
        generation_++;
        arrived_ = 0;
        cv_.notify_all();
    } else {
        cv_.wait(lock, [&] { return broken_ || generation != generation_; });
    }

    if (broken_) {
        auto error = first_error_;
        lock.unlock();
        if (error) {
            std::rethrow_exception(error);
        }
        throw std::runtime_error("rank barrier aborted");
    }
}

bool RankBarrier::all(bool value) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        all_value_ = all_value_ && value;
    }
    wait();
    bool result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        result = all_value_;
    }
    // No rank may reset the accumulator before every rank reads this round.
    wait();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        all_value_ = true;
    }
    // No rank may contribute to the next round while peers still reset.
    wait();
    return result;
}

void RankBarrier::abort(std::exception_ptr error) noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!broken_) {
            broken_ = true;
            first_error_ = error;
        }
        arrived_ = 0;
        ++generation_;
    }
    cv_.notify_all();
}

void RankBarrier::rethrow_if_broken() const {
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!broken_) {
            return;
        }
        error = first_error_;
    }
    if (error) {
        std::rethrow_exception(error);
    }
    throw std::runtime_error("rank barrier aborted");
}
} // namespace infinilm::engine
