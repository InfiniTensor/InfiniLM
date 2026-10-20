/**
 * InfiniCCL Example: Thread-per-GPU Single-Node Send/Recv
 *
 * This example transfers data from GPU 0 to GPU 1 through InfiniCCL's native
 * CCL backend without an MPI launcher.
 */

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "backend_manifest.h"
#include "infiniccl.h"
#include "utils.h"

using namespace infini::ccl;

namespace {

constexpr int kRankCount = 2;
constexpr int kSender = 0;
constexpr int kReceiver = 1;
constexpr float kSendValue = 7.0f;

struct ThreadArgs {
  int rank;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iter;
  int profile_iter;
  std::atomic_bool* all_correct;
};

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, kRankCount, args.id, args.rank));

  std::vector<float> host_buffer(args.num_elements,
                                 args.rank == kSender ? kSendValue : 0.0f);
  float* device_buffer = nullptr;
  const size_t total_bytes = args.num_elements * sizeof(float);

  CHECK_RT(Rt,
           Rt::Malloc(reinterpret_cast<void**>(&device_buffer), total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(device_buffer, host_buffer.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  auto exchange = [&]() {
    if (args.rank == kSender) {
      return infinicclSend(device_buffer, args.num_elements, infinicclFloat32,
                           kReceiver, comm, nullptr);
    }
    return infinicclRecv(device_buffer, args.num_elements, infinicclFloat32,
                         kSender, comm, nullptr);
  };

  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(exchange());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < args.profile_iter; ++i) {
    CHECK_INFINI(exchange());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed =
      timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  if (args.rank == kReceiver) {
    CHECK_RT(Rt, Rt::Memcpy(host_buffer.data(), device_buffer, total_bytes,
                            Rt::MemcpyDeviceToHost));
    const bool correct =
        Validator::ValidateResult(host_buffer.data(), args.num_elements,
                                  kSendValue, kSender, true, "Send/Recv");
    if (!correct) {
      args.all_correct->store(false, std::memory_order_relaxed);
    }
  } else {
    std::cout << "\n=== Single-Node Threaded Send/Recv Results ==="
              << std::endl;
    Metrics metrics{elapsed, total_bytes, kRankCount};
    metrics.Print();
  }

  CHECK_RT(Rt, Rt::Free(device_buffer));
  CHECK_INFINI(infinicclCommDestroy(comm));
}

}  // namespace

int main() {
  constexpr size_t kNumElements = 1 << 20;
  constexpr int kWarmupIterations = 2;
  constexpr int kProfileIterations = 20;

  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  std::atomic_bool all_correct{true};
  std::vector<std::thread> threads;
  threads.reserve(kRankCount);

  for (int rank = 0; rank < kRankCount; ++rank) {
    ThreadArgs args{rank,
                    shared_id,
                    kNumElements,
                    kWarmupIterations,
                    kProfileIterations,
                    &all_correct};
    threads.emplace_back(WorkerThread, args);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  if (!all_correct.load(std::memory_order_relaxed)) {
    std::cerr << "Send/Recv validation failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Send/Recv validation passed." << std::endl;
  return EXIT_SUCCESS;
}
