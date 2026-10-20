/**
 * InfiniCCL Example: Thread-per-GPU Single-Node AllGather
 *
 * This example validates out-of-place and in-place AllGather across two GPUs
 * through InfiniCCL's native CCL backend without an MPI launcher.
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
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
constexpr size_t kNumElements = 1 << 10;

struct ThreadArgs {
  int rank;
  infinicclUniqueId id;
  std::atomic_bool* all_correct;
};

bool Validate(const std::vector<float>& output, int rank, const char* mode) {
  for (int source = 0; source < kRankCount; ++source) {
    const float expected = static_cast<float>(source + 1);
    const size_t offset = static_cast<size_t>(source) * kNumElements;
    for (size_t i = 0; i < kNumElements; ++i) {
      if (output[offset + i] != expected) {
        std::cerr << mode << " validation failed on rank " << rank
                  << " at source rank " << source << ", element " << i
                  << ": expected " << expected << ", got " << output[offset + i]
                  << "." << std::endl;
        return false;
      }
    }
  }
  return true;
}

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, kRankCount, args.id, args.rank));

  std::vector<float> host_send(kNumElements, static_cast<float>(args.rank + 1));
  std::vector<float> host_recv(kNumElements * kRankCount, 0.0f);
  float* device_send = nullptr;
  float* device_recv = nullptr;
  const size_t send_bytes = kNumElements * sizeof(float);
  const size_t recv_bytes = send_bytes * kRankCount;

  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void**>(&device_send), send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void**>(&device_recv), recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(device_send, host_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));

  CHECK_INFINI(infinicclAllGather(device_send, device_recv, kNumElements,
                                  infinicclFloat32, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(host_recv.data(), device_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));
  if (!Validate(host_recv, args.rank, "Out-of-place AllGather")) {
    args.all_correct->store(false, std::memory_order_relaxed);
  }

  std::fill(host_recv.begin(), host_recv.end(), 0.0f);
  std::fill_n(host_recv.begin() + static_cast<size_t>(args.rank) * kNumElements,
              kNumElements, static_cast<float>(args.rank + 1));
  CHECK_RT(Rt, Rt::Memcpy(device_recv, host_recv.data(), recv_bytes,
                          Rt::MemcpyHostToDevice));

  float* local_block =
      device_recv + static_cast<size_t>(args.rank) * kNumElements;
  CHECK_INFINI(infinicclAllGather(local_block, device_recv, kNumElements,
                                  infinicclFloat32, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(host_recv.data(), device_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));
  if (!Validate(host_recv, args.rank, "In-place AllGather")) {
    args.all_correct->store(false, std::memory_order_relaxed);
  }

  CHECK_RT(Rt, Rt::Free(device_send));
  CHECK_RT(Rt, Rt::Free(device_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
}

}  // namespace

int main() {
  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  std::atomic_bool all_correct{true};
  std::vector<std::thread> threads;
  threads.reserve(kRankCount);
  for (int rank = 0; rank < kRankCount; ++rank) {
    threads.emplace_back(WorkerThread,
                         ThreadArgs{rank, shared_id, &all_correct});
  }
  for (auto& thread : threads) {
    thread.join();
  }

  if (!all_correct.load(std::memory_order_relaxed)) {
    return EXIT_FAILURE;
  }
  std::cout << "AllGather validation passed." << std::endl;
  return EXIT_SUCCESS;
}
