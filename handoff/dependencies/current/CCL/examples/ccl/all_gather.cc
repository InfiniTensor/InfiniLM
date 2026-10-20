/**
 * InfiniCCL Example: Thread-per-GPU Single-Node `AllGather`
 *
 * This example validates out-of-place and in-place `AllGather` across two GPUs
 * through InfiniCCL's native CCL backend without an MPI launcher.
 */

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "backend_manifest.h"
#include "infiniccl.h"
#include "utils.h"

using namespace infini::ccl;

namespace {

struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iter;
  int profile_iter;
  std::atomic_bool* all_correct;
};

bool Validate(const std::vector<float>& output, int size, int rank) {
  // Every rank checks the complete gathered output by source-rank block.
  const size_t num_elements = output.size() / size;
  bool correct = true;
  for (int source = 0; source < size; ++source) {
    const float expected = static_cast<float>(source + 1);
    const size_t offset = static_cast<size_t>(source) * num_elements;

    correct = Validator::ValidateResult(output.data() + offset, num_elements,
                                        expected, rank) &&
              correct;
  }
  return correct;
}

void PrintResult(const std::vector<float>& output, const char* mode, int size,
                 bool correct) {
  const char* green = "\033[32m";
  const char* red = "\033[31m";
  const char* reset = "\033[0m";
  const size_t num_elements = output.size() / size;

  std::cout << "\n=== " << mode << " AllGather Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (green + std::string("YES") + reset)
                        : (red + std::string("NO") + reset))
            << std::endl;
  std::cout << "Sample blocks: ";
  for (int source = 0; source < size; ++source) {
    const size_t offset = static_cast<size_t>(source) * num_elements;
    std::cout << "[r" << source << ": " << output[offset] << "] ";
  }
  std::cout << std::endl;

  return;
}

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  std::vector<float> host_send(args.num_elements,
                               static_cast<float>(args.rank + 1));
  std::vector<float> host_recv(args.num_elements * args.size, 0.0f);

  // Prepare separate send and receive buffers for the out-of-place case.
  float* device_send = nullptr;
  float* device_recv = nullptr;
  const size_t send_bytes = args.num_elements * sizeof(float);
  const size_t recv_bytes = send_bytes * args.size;

  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void**>(&device_send), send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void**>(&device_recv), recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(device_send, host_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  auto all_gather = [&]() {
    return infinicclAllGather(device_send, device_recv, args.num_elements,
                              infinicclFloat32, comm, nullptr);
  };

  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(all_gather());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < args.profile_iter; ++i) {
    CHECK_INFINI(all_gather());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed =
      timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  CHECK_RT(Rt, Rt::Memcpy(host_recv.data(), device_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  const bool out_of_place_correct = Validate(host_recv, args.size, args.rank);
  if (!out_of_place_correct) {
    args.all_correct->store(false, std::memory_order_relaxed);
  }
  if (args.rank == 0) {
    PrintResult(host_recv, "Out-of-place", args.size, out_of_place_correct);
  }

  // Seed only the local block, then use it as both input and output.
  std::fill(host_recv.begin(), host_recv.end(), 0.0f);
  std::fill_n(
      host_recv.begin() + static_cast<size_t>(args.rank) * args.num_elements,
      args.num_elements, static_cast<float>(args.rank + 1));
  CHECK_RT(Rt, Rt::Memcpy(device_recv, host_recv.data(), recv_bytes,
                          Rt::MemcpyHostToDevice));

  float* local_block =
      device_recv + static_cast<size_t>(args.rank) * args.num_elements;
  CHECK_INFINI(infinicclAllGather(local_block, device_recv, args.num_elements,
                                  infinicclFloat32, comm, nullptr));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(host_recv.data(), device_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));
  const bool in_place_correct = Validate(host_recv, args.size, args.rank);
  if (!in_place_correct) {
    args.all_correct->store(false, std::memory_order_relaxed);
  }

  if (args.rank == 0) {
    PrintResult(host_recv, "In-place", args.size, in_place_correct);
    std::cout << "\n=== Single-Node Threaded AllGather Results ==="
              << std::endl;
    Metrics metrics{elapsed, recv_bytes, args.size};
    metrics.Print();
  }

  // Cleanup rank-local resources.
  CHECK_RT(Rt, Rt::Free(device_send));
  CHECK_RT(Rt, Rt::Free(device_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
}

}  // namespace

int main(int argc, char** argv) {
  int num_gpus = 8;
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  int opt;
  while ((opt = getopt(argc, argv, "g:w:p:n:h")) != -1) {
    switch (opt) {
      case 'g':
        num_gpus = std::stoi(optarg);
        break;
      case 'w':
        warmup_iters = std::stoi(optarg);
        break;
      case 'p':
        profile_iters = std::stoi(optarg);
        break;
      case 'n':
        num_elements = static_cast<size_t>(std::stoull(optarg));
        break;
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  -g <num_gpus>        Number of GPUs (default: 8)\n"
                  << "  -w <warmup_iters>    Warmup iterations (default: 2)\n"
                  << "  -p <profile_iters>   Profile iterations (default: 20)\n"
                  << "  -n <num_elements>    Number of elements (default: "
                  << (1 << 20) << ")\n";
        return EXIT_SUCCESS;
      default:
        std::cerr << "Invalid argument. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  std::cout << "[Main Process] Host: " << hostname
            << " | Target GPUs: " << num_gpus << std::endl;

  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  std::atomic_bool all_correct{true};
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);

  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,         num_gpus,      shared_id,   num_elements,
                    warmup_iters, profile_iters, &all_correct};
    threads.emplace_back(WorkerThread, args);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  if (!all_correct.load(std::memory_order_relaxed)) {
    std::cerr << "AllGather validation failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "AllGather validation passed." << std::endl;
  return EXIT_SUCCESS;
}
