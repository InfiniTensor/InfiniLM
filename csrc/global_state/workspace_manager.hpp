#pragma once

#include "infinicore/tensor.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace infinilm::global_state {

class WorkspaceManager {
public:
    void reserve_slot(const std::string &name,
                      const infinicore::Shape &max_shape,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    void finalize();
    void reset_runtime_buffers();

    infinicore::Tensor get_buffer(const std::string &name,
                                  const infinicore::Shape &shape,
                                  const infinicore::DataType &dtype,
                                  const infinicore::Device &device);

    bool finalized() const noexcept { return finalized_; }
    size_t total_bytes() const noexcept { return total_bytes_; }

private:
    struct Slot {
        size_t offset{0};
        size_t capacity_bytes{0};
        infinicore::DataType dtype{infinicore::DataType::U8};
        infinicore::Device device;
    };

    static size_t aligned_nbytes(const infinicore::Shape &shape,
                                 const infinicore::DataType &dtype);

    bool finalized_{false};
    size_t total_bytes_{0};
    infinicore::Tensor storage_;
    std::unordered_map<std::string, Slot> slots_;
    std::unordered_map<std::string, infinicore::Tensor> runtime_views_;
};

} // namespace infinilm::global_state
