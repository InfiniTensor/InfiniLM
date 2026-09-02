#include "communication_group.hpp"
#include "tcp_rendezvous.hpp"

#include <spdlog/spdlog.h>

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

void initializeCommunicators(infinicore::Device::Type device_type,
                             const std::vector<int> &device_ids,
                             int world_size,
                             const infinicclUniqueId &unique_id,
                             int rank_offset,
                             std::vector<infinicclComm_t> &communicators) {
    const size_t local_size = device_ids.size();
    if (communicators.size() != local_size) {
        throw std::logic_error("communicator and device counts do not match");
    }

    std::vector<std::exception_ptr> errors(local_size);
    std::vector<std::thread> workers;
    workers.reserve(local_size);
    std::mutex start_mutex;
    std::condition_variable start_cv;
    bool start = false;
    bool cancel = false;
    try {
        for (size_t local_rank = 0; local_rank < local_size; ++local_rank) {
            workers.emplace_back([&, local_rank] {
                {
                    std::unique_lock<std::mutex> lock(start_mutex);
                    start_cv.wait(lock, [&] { return start || cancel; });
                    if (cancel) {
                        return;
                    }
                }
                try {
                    infinicore::context::setDevice(
                        infinicore::Device(device_type, device_ids[local_rank]));
                    checkInfiniccl(
                        "infinicclCommInitRank",
                        infinicclCommInitRank(
                            &communicators[local_rank],
                            world_size,
                            unique_id,
                            rank_offset + static_cast<int>(local_rank)));
                } catch (...) {
                    errors[local_rank] = std::current_exception();
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
        destroyCommunicators(device_type, device_ids, communicators);
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
            destroyCommunicators(device_type, device_ids, communicators);
            std::rethrow_exception(error);
        }
    }
}

} // namespace

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config,
                                       infinicore::Device::Type device_type)
    : dist_config_(dist_config),
      device_type_(device_type),
      communicators_(dist_config.tp_device_ids.size(), nullptr),
      world_communicators_(dist_config.tp_device_ids.size(), nullptr) {
    if (dist_config_.pp_size < 1) {
        throw std::invalid_argument("pipeline parallel size must be at least 1");
    }
    if (dist_config_.pp_stage < 0 || dist_config_.pp_stage >= dist_config_.pp_size) {
        throw std::invalid_argument("pipeline parallel stage must be in [0, pp_size)");
    }

    const size_t tp_size = dist_config_.tp_device_ids.size();
    const size_t rank_limit = static_cast<size_t>(std::numeric_limits<int>::max());
    if (tp_size == 0) {
        throw std::invalid_argument("tensor parallel device list must not be empty");
    }
    if (tp_size > rank_limit) {
        throw std::invalid_argument("tensor parallel world size exceeds the InfiniCCL rank limit");
    }
    if (tp_size > rank_limit / static_cast<size_t>(dist_config_.pp_size)) {
        throw std::invalid_argument("combined tensor and pipeline parallel world size exceeds the InfiniCCL rank limit");
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

    if (tp_size > 1) {
        if (device_type_ == infinicore::Device::Type::kCambricon) {
            checkInfiniccl(
                "infinicclCommInitAll",
                infinicclCommInitAll(
                    communicators_.data(),
                    static_cast<int>(tp_size),
                    dist_config_.tp_device_ids.data()));
        } else {
            infinicclUniqueId unique_id{};
            checkInfiniccl("infinicclGetUniqueId", infinicclGetUniqueId(&unique_id));
            initializeCommunicators(
                device_type_,
                dist_config_.tp_device_ids,
                static_cast<int>(tp_size),
                unique_id,
                0,
                communicators_);
        }
        spdlog::info(
            "Intra-node TP communicator established: node_rank={}, local_ranks={}",
            dist_config_.pp_stage,
            tp_size);
    }

    if (dist_config_.pp_size > 1) {
        try {
            infinicclUniqueId unique_id{};
            if (dist_config_.pp_stage == 0) {
                checkInfiniccl("infinicclGetUniqueId", infinicclGetUniqueId(&unique_id));
            }
            broadcast_rendezvous_payload(
                TcpRendezvousConfig{
                    dist_config_.master_addr,
                    dist_config_.master_port,
                    dist_config_.pp_size,
                    dist_config_.pp_stage,
                },
                &unique_id,
                sizeof(unique_id));

            const int local_tp_size = static_cast<int>(tp_size);
            const int pp_world_size = dist_config_.pp_size * local_tp_size;
            initializeCommunicators(
                device_type_,
                dist_config_.tp_device_ids,
                pp_world_size,
                unique_id,
                dist_config_.pp_stage * local_tp_size,
                world_communicators_);
            spdlog::info(
                "Global InfiniCCL communicator established: role={}, node_rank={}, nodes={}, local_tp_ranks={}, world_size={}",
                dist_config_.pp_stage == 0 ? "coordinator" : "participant",
                dist_config_.pp_stage,
                dist_config_.pp_size,
                local_tp_size,
                pp_world_size);
        } catch (...) {
            destroyCommunicators(
                device_type_, dist_config_.tp_device_ids, world_communicators_);
            destroyCommunicators(
                device_type_, dist_config_.tp_device_ids, communicators_);
            throw;
        }
    }
}

const DistConfig &CommunicationGroup::get_dist_config() const {
    return dist_config_;
}

RankInfo CommunicationGroup::get_rank_info(int rank) const {
    if (rank < 0 || static_cast<size_t>(rank) >= dist_config_.tp_device_ids.size()) {
        throw std::out_of_range("tensor parallel rank " + std::to_string(rank)
                                + " is out of range");
    }

    RankInfo info(infinicore::Device(device_type_, dist_config_.tp_device_ids[rank]));
    info.tp_size = static_cast<int>(dist_config_.tp_device_ids.size());
    info.tp_rank = rank;
    info.comm = communicators_[rank];
    info.pp_size = dist_config_.pp_size;
    info.pp_stage = dist_config_.pp_stage;
    info.world_size = dist_config_.pp_size * info.tp_size;
    info.world_rank = dist_config_.pp_stage * info.tp_size + info.tp_rank;
    info.world_comm = world_communicators_[rank];
    return info;
}

int CommunicationGroup::get_world_size() const {
    return static_cast<int>(dist_config_.tp_device_ids.size());
}

CommunicationGroup::~CommunicationGroup() {
    destroyCommunicators(device_type_, dist_config_.tp_device_ids, world_communicators_);
    destroyCommunicators(device_type_, dist_config_.tp_device_ids, communicators_);
}

} // namespace infinilm::engine::distributed
