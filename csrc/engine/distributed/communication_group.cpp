#include "communication_group.hpp"

#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

namespace infinilm::engine::distributed {
namespace {

void checkInfiniccl(const char *operation, infinicclResult_t result) {
    if (result == infinicclSuccess) {
        return;
    }
    throw std::runtime_error("InfiniCCL operation `" + std::string(operation)
                             + "` failed with result " + std::to_string(static_cast<int>(result)));
}

void destroyCommunicators(infinicore::Device::Type device_type,
                          const std::vector<int> &device_ids,
                          std::vector<infinicclComm_t> &communicators) noexcept {
    infinicore::Device previous_device;
    bool restore_device = false;
    try {
        previous_device = infinicore::context::getDevice();
        restore_device = true;
    } catch (...) {
    }

    for (size_t rank = 0; rank < communicators.size(); ++rank) {
        if (communicators[rank] == nullptr) {
            continue;
        }
        try {
            infinicore::context::setDevice(infinicore::Device(device_type, device_ids[rank]));
            (void)infinicclCommDestroy(communicators[rank]);
        } catch (...) {
        }
        communicators[rank] = nullptr;
    }

    if (restore_device) {
        try {
            infinicore::context::setDevice(previous_device);
        } catch (...) {
        }
    }
}

} // namespace

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config, infinicore::Device::Type device_type)
    : dist_config_(dist_config), device_type_(device_type),
      communicators_(dist_config.tp_device_ids.size(), nullptr) {
    const size_t world_size = dist_config_.tp_device_ids.size();
    if (world_size == 0) {
        throw std::invalid_argument("tensor parallel device list must not be empty");
    }
    if (world_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("tensor parallel world size exceeds the InfiniCCL rank limit");
    }

    const size_t device_count = infinicore::context::getDeviceCount(device_type_);
    std::unordered_set<int> unique_device_ids;
    for (int device_id : dist_config_.tp_device_ids) {
        if (device_id < 0 || static_cast<size_t>(device_id) >= device_count) {
            throw std::invalid_argument("tensor parallel device ID " + std::to_string(device_id)
                                        + " is outside the available range [0, "
                                        + std::to_string(device_count) + ")");
        }
        if (!unique_device_ids.insert(device_id).second) {
            throw std::invalid_argument("tensor parallel device ID " + std::to_string(device_id)
                                        + " is duplicated");
        }
    }

    if (world_size == 1) {
        return;
    }

    infinicclUniqueId unique_id{};
    checkInfiniccl("infinicclGetUniqueId", infinicclGetUniqueId(&unique_id));

    std::vector<std::exception_ptr> errors(world_size);
    std::vector<std::thread> workers;
    workers.reserve(world_size);
    std::mutex start_mutex;
    std::condition_variable start_cv;
    bool start = false;
    bool cancel = false;
    try {
        for (size_t rank = 0; rank < world_size; ++rank) {
            workers.emplace_back([&, rank] {
                {
                    std::unique_lock<std::mutex> lock(start_mutex);
                    start_cv.wait(lock, [&] { return start || cancel; });
                    if (cancel) {
                        return;
                    }
                }
                try {
                    infinicore::context::setDevice(
                        infinicore::Device(device_type_, dist_config_.tp_device_ids[rank]));
                    checkInfiniccl(
                        "infinicclCommInitRank",
                        infinicclCommInitRank(&communicators_[rank],
                                              static_cast<int>(world_size),
                                              unique_id,
                                              static_cast<int>(rank)));
                } catch (...) {
                    errors[rank] = std::current_exception();
                }
            });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(start_mutex);
            cancel = true;
        }
        start_cv.notify_all();
        for (auto &worker : workers) {
            worker.join();
        }
        destroyCommunicators(device_type_, dist_config_.tp_device_ids, communicators_);
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(start_mutex);
        start = true;
    }
    start_cv.notify_all();
    for (auto &worker : workers) {
        worker.join();
    }
    for (const auto &error : errors) {
        if (error) {
            destroyCommunicators(device_type_, dist_config_.tp_device_ids, communicators_);
            std::rethrow_exception(error);
        }
    }
}

const DistConfig &CommunicationGroup::get_dist_config() const {
    return dist_config_;
}

RankInfo CommunicationGroup::get_rank_info(int rank) const {
    if (rank < 0 || static_cast<size_t>(rank) >= dist_config_.tp_device_ids.size()) {
        throw std::out_of_range("tensor parallel rank " + std::to_string(rank) + " is out of range");
    }

    RankInfo info(infinicore::Device(device_type_, dist_config_.tp_device_ids[rank]));
    info.tp_size = static_cast<int>(dist_config_.tp_device_ids.size());
    info.tp_rank = rank;
    info.comm = communicators_[rank];
    return info;
}

int CommunicationGroup::get_world_size() const {
    return static_cast<int>(dist_config_.tp_device_ids.size());
}

CommunicationGroup::~CommunicationGroup() {
    destroyCommunicators(device_type_, dist_config_.tp_device_ids, communicators_);
}

} // namespace infinilm::engine::distributed
