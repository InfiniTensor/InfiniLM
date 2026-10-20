#include "execution/submit.hpp"
#include "kimi_reduce_precision_r3226.hpp"
#include "communication_group.hpp"
#include "execution/graph/graph.hpp"
#include "runtime/runtime.hpp"
#include "tcp_rendezvous.hpp"
#include <infini/rt/device_guard.h>

#include <infini/rt.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

namespace infinilm::engine::distributed {
namespace {

using infinilm::runtime::Runtime;
using infinilm::runtime::Tensor;

infinicclDataType_t cclDataType(infinilm::runtime::DataType dtype) {
    switch (dtype) {
    case infinilm::runtime::DataType::kInt8:
        return infinicclInt8;
    case infinilm::runtime::DataType::kInt16:
        return infinicclInt16;
    case infinilm::runtime::DataType::kInt32:
        return infinicclInt32;
    case infinilm::runtime::DataType::kInt64:
        return infinicclInt64;
    case infinilm::runtime::DataType::kUInt8:
        return infinicclUInt8;
    case infinilm::runtime::DataType::kUInt16:
        return infinicclUInt16;
    case infinilm::runtime::DataType::kUInt32:
        return infinicclUInt32;
    case infinilm::runtime::DataType::kUInt64:
        return infinicclUInt64;
    case infinilm::runtime::DataType::kFloat16:
        return infinicclFloat16;
    case infinilm::runtime::DataType::kBFloat16:
        return infinicclBFloat16;
    case infinilm::runtime::DataType::kFloat32:
        return infinicclFloat32;
    case infinilm::runtime::DataType::kFloat64:
        return infinicclFloat64;
    }
    throw std::invalid_argument("unsupported data type for InfiniCCL");
}

void requireSubmissionPolicy(const Runtime &runtime) {
    runtime.assertOwnerThread();
    if (!runtime.ownerDrainFailStopEnabled()) {
        throw std::logic_error("InfiniCCL submission requires owner drain fail-stop policy");
    }
}

void validateTensor(Runtime &runtime, const Tensor &tensor, const char *operation) {
    runtime.assertOwnerThread();
    if (!tensor || !tensor->is_contiguous()) {
        throw std::invalid_argument(std::string(operation) + " requires a contiguous tensor");
    }
    if (runtime.device() != tensor->device()) {
        throw std::invalid_argument(std::string(operation) + " runtime and tensor devices differ");
    }
    (void)cclDataType(tensor->dtype());
}

void validatePair(Runtime &runtime, const Tensor &output, const Tensor &input,
                  const char *operation) {
    validateTensor(runtime, output, operation);
    validateTensor(runtime, input, operation);
    if (output->dtype() != input->dtype()) {
        throw std::invalid_argument(std::string(operation) + " tensor dtypes differ");
    }
}

void requireCommunicator(infinicclComm_t communicator) {
    if (communicator == nullptr) {
        throw std::invalid_argument("InfiniCCL submission requires a communicator");
    }
}

void validateUniformSplits(Runtime &runtime, const Tensor &output,
                           const Tensor &input, const std::vector<size_t> &splits,
                           bool gather) {
    validatePair(runtime, output, input, gather ? "AllGather" : "ReduceScatter");
    const auto &small = gather ? input : output;
    const auto &large = gather ? output : input;
    if (splits.empty() || small->ndim() == 0 || small->shape()[0] == 0
        || large->ndim() != small->ndim()) {
        throw std::invalid_argument("InfiniCCL uniform split metadata is invalid");
    }
    if (!std::all_of(splits.begin(), splits.end(), [&](size_t size) {
            return size == small->shape()[0];
        })) {
        throw std::runtime_error(gather
                                     ? "InfiniCCL does not support variable-count all-gather"
                                     : "InfiniCCL does not support variable-count reduce-scatter");
    }
    if (small->shape()[0] > std::numeric_limits<size_t>::max() / splits.size()
        || large->shape()[0] != small->shape()[0] * splits.size()
        || small->numel() > std::numeric_limits<size_t>::max() / splits.size()
        || large->numel() != small->numel() * splits.size()) {
        throw std::invalid_argument("InfiniCCL uniform split sizes do not match tensors");
    }
    for (size_t dim = 1; dim < small->ndim(); ++dim) {
        if (small->shape()[dim] != large->shape()[dim]) {
            throw std::invalid_argument("InfiniCCL uniform split tensor shapes differ");
        }
    }
}

// Native arguments are materialized before entering this fail-stop region.
// A provider error/exception may have submitted work on only some ranks;
// a host barrier, local exception unwind or stream drain cannot abort peers.
template <typename NativeCall>
void submitNow(Runtime &runtime, Tensor output, const Tensor &input,
               const NativeCall &native_call) {
    requireSubmissionPolicy(runtime);
    infini::rt::DeviceGuard guard{runtime.device().type(), runtime.device().index()};
    if (!guard.active()) {
        throw std::runtime_error("InfiniCCL could not select its worker device");
    }
    const void *input_data = input ? input->data() : nullptr;
    void *output_data = output ? output->data() : nullptr;
    void *stream = reinterpret_cast<void *>(runtime.stream());
    infinicclResult_t result;
    try {
        result = native_call(input_data, output_data, stream);
    } catch (...) {
        std::_Exit(86);
    }
    if (result != infinicclSuccess) {
        std::_Exit(86);
    }
}

// This only handles existing graph ownership/submission. It has no operation
// key, dispatcher, instance/cache, derived operator or computation policy.
template <typename NativeCall>
void submitCollective(Runtime &runtime, Tensor output, const Tensor &input,
                      bool capture_safe, NativeCall native_call) {
    requireSubmissionPolicy(runtime);
    if (!runtime.isGraphRecording()) {
        submitNow(runtime, output, input, native_call);
        return;
    }
    std::optional<infinilm::execution::graph::GraphTensor> recorded_output;
    std::optional<infinilm::execution::graph::GraphTensor> recorded_input;
    if (output) {
        recorded_output.emplace(runtime, output);
    }
    if (input) {
        recorded_input.emplace(runtime, input);
    }
    runtime.addGraphOperator(std::make_shared<infinilm::execution::graph::RecordedCall>(
        [runtime_owner = runtime.shared_from_this(),
         recorded_output = std::move(recorded_output),
         recorded_input = std::move(recorded_input),
         native_call = std::move(native_call)] {
            submitNow(*runtime_owner,
                      recorded_output ? static_cast<const Tensor &>(*recorded_output) : Tensor{},
                      recorded_input ? static_cast<const Tensor &>(*recorded_input) : Tensor{},
                      native_call);
        },
        capture_safe));
}

template <infini::rt::Device::Type... types>
bool backendEnabled(infini::rt::Device::Type type, infini::rt::List<types...>) {
    return ((type == types && infini::rt::DeviceEnabled<types>::value) || ...);
}

void checkInfiniccl(const char *operation, infinicclResult_t result) {
    if (result == infinicclSuccess) {
        return;
    }
    throw std::runtime_error("InfiniCCL operation `" + std::string(operation)
                             + "` failed with result " + std::to_string(static_cast<int>(result)));
}

void destroyCommunicators(infinilm::runtime::Device::Type device_type,
                          const std::vector<int> &device_ids,
                          std::vector<infinicclComm_t> &communicators) noexcept {
    for (size_t rank = 0; rank < communicators.size(); ++rank) {
        if (communicators[rank] == nullptr) {
            continue;
        }
        try {
            infini::rt::DeviceGuard device_guard(device_type, device_ids[rank], true);
            try {
                if (infinicclCommDestroy(communicators[rank]) != infinicclSuccess) {
                    std::_Exit(86);
                }
            } catch (...) {
                std::_Exit(86);
            }
            communicators[rank] = nullptr;
        } catch (...) {
            std::_Exit(86);
        }
    }
}

void initializeCommunicators(infinilm::runtime::Device::Type device_type,
                             const std::vector<int> &device_ids,
                             int world_size,
                             const infinicclUniqueId &unique_id,
                             int rank_offset,
                             std::vector<infinicclComm_t> &communicators) {
    const size_t local_size = device_ids.size();
    if (communicators.size() != local_size) {
        throw std::logic_error("communicator and device counts do not match");
    }

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
                    // InfiniCCL only requires the native device to be current. Avoid creating
                    // thread-local InfiniCore runtimes in these short-lived initialization threads.
                    infini::rt::set_runtime_device_type(device_type);
                    const auto set_device_status = infini::rt::runtime::SetDevice(device_ids[local_rank]);
                    if (set_device_status != infini::rt::runtime::kSuccess) {
                        std::_Exit(86);
                    }
                    if (infinicclCommInitRank(
                            &communicators[local_rank],
                            world_size,
                            unique_id,
                            rank_offset + static_cast<int>(local_rank))
                        != infinicclSuccess) {
                        std::_Exit(86);
                    }
                } catch (...) {
                    std::_Exit(86);
                }
            });
        }
    } catch (...) {
        if (workers.empty()) {
            throw;
        }
        try {
            {
                std::lock_guard<std::mutex> lock(start_mutex);
                cancel = true;
            }
            start_cv.notify_all();
            for (auto &worker : workers) {
                worker.join();
            }
        } catch (...) {
            std::_Exit(86);
        }
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(start_mutex);
        start = true;
    }
    start_cv.notify_all();
    try {
        for (auto &worker : workers) {
            worker.join();
        }
    } catch (...) {
        std::_Exit(86);
    }
}

} // namespace

void allreduce_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                const infinilm::runtime::Tensor &input, infinicclRedOp_t op,
                infinicclComm_t communicator) {
    validatePair(runtime, output, input, "AllReduce");
    if (output->numel() != input->numel()) {
        throw std::invalid_argument("AllReduce tensor element counts differ");
    }
    requireCommunicator(communicator);
    requireSubmissionPolicy(runtime);
    if(input->dtype()==infinilm::runtime::DataType::kBFloat16&&op==infinicclSum&&reduce_precision_r3226::enabled(runtime)){
        infinilm::runtime::Tensor wide;
        try{
            wide=infinilm::runtime::Tensor::empty(runtime,input->shape(),infinilm::runtime::DataType::kFloat32);
            infinilm::execution::submit<infini::ops::Cast>(runtime,infini::ops::Cast::DefaultConfig(runtime.device().type()),input,wide);
            allreduce_(runtime,wide,wide,op,communicator);
            infinilm::execution::submit<infini::ops::Cast>(runtime,infini::ops::Cast::DefaultConfig(runtime.device().type()),wide,output);
            runtime.syncStream(); // Initial correctness candidate retains scratch through completion.
            reduce_precision_r3226::record(runtime,wide->nbytes());
            INFINILM_TRACE_OP(AllReduce,output);return;
        }catch(...){try{runtime.syncStream();}catch(...){std::terminate();}throw;}
    }
    const auto dtype = cclDataType(input->dtype());
    const auto count = input->numel();
    submitCollective(runtime, output, input, true,
                     [count, dtype, op, communicator](const void *src, void *dst, void *stream) {
                         return infinicclAllReduce(src, dst, count, dtype, op, communicator, stream);
                     });
    INFINILM_TRACE_OP(AllReduce, output);
}

void allgather_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                const infinilm::runtime::Tensor &input, infinicclComm_t communicator) {
    validatePair(runtime, output, input, "AllGather");
    if (input->numel() == 0 || output->numel() == 0
        || output->numel() % input->numel() != 0) {
        throw std::invalid_argument("AllGather tensor element counts are invalid");
    }
    requireCommunicator(communicator);
    const auto dtype = cclDataType(input->dtype());
    const auto count = input->numel();
    submitCollective(runtime, output, input, false,
                     [count, dtype, communicator](const void *src, void *dst, void *stream) {
                         return infinicclAllGather(src, dst, count, dtype, communicator, stream);
                     });
    INFINILM_TRACE_OP(AllGather, output);
}

void reduce_scatter_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                     const infinilm::runtime::Tensor &input, infinicclRedOp_t op,
                     infinicclComm_t communicator) {
    validatePair(runtime, output, input, "ReduceScatter");
    if (input->numel() == 0 || output->numel() == 0
        || input->numel() % output->numel() != 0) {
        throw std::invalid_argument("ReduceScatter tensor element counts are invalid");
    }
    requireCommunicator(communicator);
    const auto dtype = cclDataType(input->dtype());
    const auto count = output->numel();
    submitCollective(runtime, output, input, false,
                     [count, dtype, op, communicator](const void *src, void *dst, void *stream) {
                         return infinicclReduceScatter(src, dst, count, dtype, op, communicator, stream);
                     });
    INFINILM_TRACE_OP(ReduceScatter, output);
}

void send(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input,
          int peer, infinicclComm_t communicator) {
    validateTensor(runtime, input, "Send");
    if (input->numel() == 0) {
        throw std::invalid_argument("Send requires a nonempty tensor");
    }
    requireCommunicator(communicator);
    const auto dtype = cclDataType(input->dtype());
    const auto count = input->numel();
    submitCollective(runtime, {}, input, false,
                     [count, dtype, peer, communicator](const void *src, void *, void *stream) {
                         return infinicclSend(src, count, dtype, peer, communicator, stream);
                     });
}

void recv_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
           int peer, infinicclComm_t communicator) {
    validateTensor(runtime, output, "Recv");
    if (output->numel() == 0) {
        throw std::invalid_argument("Recv requires a nonempty tensor");
    }
    requireCommunicator(communicator);
    const auto dtype = cclDataType(output->dtype());
    const auto count = output->numel();
    submitCollective(runtime, output, {}, false,
                     [count, dtype, peer, communicator](const void *, void *dst, void *stream) {
                         return infinicclRecv(dst, count, dtype, peer, communicator, stream);
                     });
}

void allgatherv_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                 const infinilm::runtime::Tensor &input, const std::vector<size_t> &split_sizes,
                 infinicclComm_t communicator) {
    validateUniformSplits(runtime, output, input, split_sizes, true);
    allgather_(runtime, output, input, communicator);
}

void allgatherv_many_(infinilm::runtime::Runtime &runtime,
                      const std::vector<infinilm::runtime::Tensor> &outputs,
                      const std::vector<infinilm::runtime::Tensor> &inputs,
                      const std::vector<size_t> &split_sizes,
                      infinicclComm_t communicator) {
    runtime.assertOwnerThread();
    if (outputs.size() != inputs.size() || split_sizes.empty()) {
        throw std::invalid_argument("AllGather batch or split metadata is invalid");
    }
    if (inputs.empty()) {
        return;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        validateUniformSplits(runtime, outputs[i], inputs[i], split_sizes, true);
    }
    requireCommunicator(communicator);
    requireSubmissionPolicy(runtime);
    for (size_t i = 0; i < inputs.size(); ++i) {
        allgather_(runtime, outputs[i], inputs[i], communicator);
    }
}

void reduce_scatterv_(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor output,
                      const infinilm::runtime::Tensor &input, const std::vector<size_t> &split_sizes,
                      infinicclRedOp_t op, infinicclComm_t communicator) {
    validateUniformSplits(runtime, output, input, split_sizes, false);
    reduce_scatter_(runtime, output, input, op, communicator);
}

void reduce_scatterv_many_(infinilm::runtime::Runtime &runtime,
                           const std::vector<infinilm::runtime::Tensor> &outputs,
                           const std::vector<infinilm::runtime::Tensor> &inputs,
                           const std::vector<size_t> &split_sizes,
                           infinicclRedOp_t op, infinicclComm_t communicator) {
    runtime.assertOwnerThread();
    if (outputs.size() != inputs.size() || split_sizes.empty()) {
        throw std::invalid_argument("ReduceScatter batch or split metadata is invalid");
    }
    if (inputs.empty()) {
        return;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        validateUniformSplits(runtime, outputs[i], inputs[i], split_sizes, false);
    }
    requireCommunicator(communicator);
    requireSubmissionPolicy(runtime);
    for (size_t i = 0; i < inputs.size(); ++i) {
        reduce_scatter_(runtime, outputs[i], inputs[i], op, communicator);
    }
}

CommunicationGroup::CommunicationGroup(const DistConfig &dist_config,
                                       infinilm::runtime::Device::Type device_type)
    : dist_config_(dist_config),
      device_type_(device_type),
      topology_(dist_config.tp_device_ids.size(), dist_config.pp_size, dist_config.pp_stage),
      communicators_(dist_config.tp_device_ids.size(), nullptr),
      world_communicators_(dist_config.tp_device_ids.size(), nullptr) {
    const size_t tp_size = static_cast<size_t>(topology_.tp_size());

    if (!backendEnabled(device_type_, infini::rt::AllDeviceTypes{})) {
        throw std::invalid_argument("communication device backend is not enabled in InfiniRT");
    }
    // Bootstrap and teardown only need the native device; neither creates an LM Runtime.
    infini::rt::DeviceGuard device_guard(device_type_, 0, true);
    int device_count = 0;
    const auto count_status = infini::rt::runtime::GetDeviceCount(&device_count);
    if (count_status != infini::rt::runtime::kSuccess || device_count < 0) {
        throw std::runtime_error("InfiniRT device enumeration failed");
    }
    std::unordered_set<int> unique_device_ids;
    for (int device_id : dist_config_.tp_device_ids) {
        if (device_id < 0 || device_id >= device_count) {
            throw std::invalid_argument("tensor parallel device ID " + std::to_string(device_id)
                                        + " is outside the available range [0, "
                                        + std::to_string(device_count) + ")");
        }
        if (!unique_device_ids.insert(device_id).second) {
            throw std::invalid_argument("tensor parallel device ID " + std::to_string(device_id)
                                        + " is duplicated");
        }
    }

    try {
        if (tp_size > 1) {
            const bool use_single_node_init_all = device_type_ == infinilm::runtime::Device::Type::kCambricon
                                               || device_type_ == infinilm::runtime::Device::Type::kAscend;
            if (use_single_node_init_all) {
                try {
                    if (infinicclCommInitAll(
                            communicators_.data(),
                            static_cast<int>(tp_size),
                            dist_config_.tp_device_ids.data())
                        != infinicclSuccess) {
                        std::_Exit(86);
                    }
                } catch (...) {
                    std::_Exit(86);
                }
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

            const int local_tp_size = topology_.tp_size();
            const int pp_world_size = topology_.world_size();
            initializeCommunicators(
                device_type_,
                dist_config_.tp_device_ids,
                pp_world_size,
                unique_id,
                topology_.world_rank(0),
                world_communicators_);
            spdlog::info(
                "Global InfiniCCL communicator established: role={}, node_rank={}, nodes={}, local_tp_ranks={}, world_size={}",
                dist_config_.pp_stage == 0 ? "coordinator" : "participant",
                dist_config_.pp_stage,
                dist_config_.pp_size,
                local_tp_size,
                pp_world_size);
        }
    } catch (...) {
        destroyCommunicators(
            device_type_, dist_config_.tp_device_ids, world_communicators_);
        destroyCommunicators(
            device_type_, dist_config_.tp_device_ids, communicators_);
        throw;
    }
}

const DistConfig &CommunicationGroup::get_dist_config() const {
    return dist_config_;
}

RankInfo CommunicationGroup::get_rank_info(int rank) const {
    const int world_rank = topology_.world_rank(rank);

    RankInfo info(infinilm::runtime::Device(device_type_, dist_config_.tp_device_ids[rank]));
    info.tp_size = topology_.tp_size();
    info.tp_rank = rank;
    info.comm = communicators_[rank];
    info.pp_size = topology_.pp_size();
    info.pp_stage = topology_.pp_stage();
    info.world_size = topology_.world_size();
    info.world_rank = world_rank;
    info.world_comm = world_communicators_[rank];
    return info;
}

int CommunicationGroup::tp_size() const {
    return topology_.tp_size();
}

CommunicationGroup::~CommunicationGroup() {
    destroyCommunicators(device_type_, dist_config_.tp_device_ids, world_communicators_);
    destroyCommunicators(device_type_, dist_config_.tp_device_ids, communicators_);
}

} // namespace infinilm::engine::distributed

#include "kimi_reduce_precision_r3226.inc"
