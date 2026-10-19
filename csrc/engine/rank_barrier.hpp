#pragma once

#include <condition_variable>
#include <exception>
#include <mutex>

namespace infinilm::engine {
class RankBarrier {
public:
    explicit RankBarrier(size_t nranks);

    void wait();
    bool all(bool value);
    void abort(std::exception_ptr error) noexcept;
    void rethrow_if_broken() const;

private:
    const size_t thread_count_;
    size_t arrived_ = 0;
    size_t generation_ = 0;
    bool broken_ = false;
    bool all_value_ = true;
    std::exception_ptr first_error_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};
} // namespace infinilm::engine
