/**
 * InfiniCCL Example: Thread-per-GPU Single-Node `ReduceScatter`
 *
 * This example spawns one CPU thread per GPU on a single node, generates a
 * shared UniqueID, and performs `ReduceScatter` entirely through InfiniCCL's
 * native CCL backend without any OpenMPI dependency.
 *
 * Every rank's send buffer is split into `size` equally sized blocks; block `b`
 * of rank `r` holds `(r + 1) + b`. After the reduction, rank `r` receives block
 * `r` of the element-wise sum, so each block value is validated independently
 * to catch misplaced destination chunks.
 *
 * Note: to properly run this example, you should specify `--launcher none`
 * since it requires no MPI process spawning.
 */

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Public API
#include "infiniccl.h"

// Example-Specific Utilities
#include "utils.h"

// Internal Headers (Accessible via example-specific include paths, technically
// not public APIs)
#include "backend_manifest.h"

using namespace infini::ccl;

namespace {

// Structure to pass execution data to each GPU worker thread.
struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t recv_count;  // Elements received per rank.
  int warmup_iter;
  int profile_iter;
  std::atomic_bool* all_correct;
};

// Value stored in block `block` of rank `rank`'s send buffer. The block index
// is embedded in the payload so a block mixed up by the backend cannot pass
// validation.
float SeedValue(int rank, size_t block) {
  return static_cast<float>(rank + 1) + static_cast<float>(block);
}

// Value block `block` holds after the element-wise sum over all ranks.
float ExpectedValue(int size, size_t block) {
  const float rank_sum = static_cast<float>(size * (size + 1) / 2);
  return rank_sum + static_cast<float>(size) * static_cast<float>(block);
}

// Every rank checks the single reduced block it received from the collective.
bool Validate(const std::vector<float>& output, int size, int rank) {
  return Validator::ValidateResult(
      output.data(), output.size(),
      ExpectedValue(size, static_cast<size_t>(rank)), rank);
}

void PrintResult(const std::vector<float>& output, int rank, int size,
                 bool correct) {
  const char* green = "\033[32m";
  const char* red = "\033[31m";
  const char* reset = "\033[0m";

  std::cout << "\n=== ReduceScatter Results ===" << std::endl;
  std::cout << "Rank " << rank << " owns block " << rank << std::endl;
  std::cout << "Correct: "
            << (correct ? (green + std::string("YES") + reset)
                        : (red + std::string("NO") + reset))
            << std::endl;
  // Sample the first element of the received block; the rest is validated
  // above.
  std::cout << "Sample:  " << output.front() << " (expected "
            << ExpectedValue(size, static_cast<size_t>(rank)) << ")"
            << std::endl;
}

void PrintMetrics(double elapsed_ms, size_t send_bytes, int size) {
  const double seconds = elapsed_ms / 1000.0;
  const double gigabytes =
      static_cast<double>(send_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double alg_bw = gigabytes / seconds;
  // `ReduceScatter` moves `(n - 1) / n` of the send buffer per rank.
  const double bus_bw =
      (static_cast<double>(size - 1) / static_cast<double>(size)) * alg_bw;

  std::cout << "Time:           " << std::fixed << std::setprecision(3)
            << elapsed_ms << " ms" << std::endl;
  std::cout << "Throughput:     " << std::fixed << std::setprecision(2)
            << bus_bw << " GB/s (Bus BW)" << std::endl;
  std::cout << "Alg Bandwidth:  " << std::fixed << std::setprecision(2)
            << alg_bw << " GB/s" << std::endl;
  return;
}

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  // Bind this specific CPU thread to its designated local GPU device.
  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  // The send buffer holds one block per rank, the receive buffer one block.
  const size_t recv_bytes = args.recv_count * sizeof(float);
  const size_t send_bytes = recv_bytes * static_cast<size_t>(args.size);

  std::vector<float> host_send(send_bytes / sizeof(float));
  std::vector<float> host_recv(args.recv_count, 0.0f);
  for (size_t block = 0; block < static_cast<size_t>(args.size); ++block) {
    std::fill_n(host_send.begin() + block * args.recv_count, args.recv_count,
                SeedValue(args.rank, block));
  }

  // Allocate the device buffers once and keep the data resident for the whole
  // run: the timed loop never touches the host.
  float* device_send = nullptr;
  float* device_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void**>(&device_send), send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void**>(&device_recv), recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(device_send, host_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  auto reduce_scatter = [&]() {
    return infinicclReduceScatter(device_send, device_recv, args.recv_count,
                                  infinicclFloat32, infinicclSum, comm,
                                  nullptr);
  };

  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(reduce_scatter());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // All launches are enqueued back to back and the stream is synchronized only
  // once, so the measurement reflects the collective itself rather than the
  // per-iteration launch latency.
  Timer timer;
  for (int i = 0; i < args.profile_iter; ++i) {
    CHECK_INFINI(reduce_scatter());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed =
      timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  CHECK_RT(Rt, Rt::Memcpy(host_recv.data(), device_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  const bool correct = Validate(host_recv, args.size, args.rank);
  if (!correct) {
    args.all_correct->store(false, std::memory_order_relaxed);
  }

  if (args.rank == 0) {
    PrintResult(host_recv, args.rank, args.size, correct);
    std::cout << "\n=== Single-Node Threaded ReduceScatter Results ==="
              << std::endl;
    std::cout << "Receive size: " << args.recv_count << " floats per rank ("
              << recv_bytes / 1024 / 1024
              << " MB), send size: " << send_bytes / 1024 / 1024 << " MB"
              << std::endl;
    PrintMetrics(elapsed, send_bytes, args.size);
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
  size_t recv_count = 1 << 20;

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
        recv_count = static_cast<size_t>(std::stoull(optarg));
        break;
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  -g <num_gpus>     Number of GPUs (default: 8)\n"
                  << "  -w <warmup_iters> Warmup iterations (default: 2)\n"
                  << "  -p <profile_iters> Profile iterations (default: 20)\n"
                  << "  -n <recv_count>   Elements received per rank "
                     "(default: "
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
    ThreadArgs args{rank,         num_gpus,      shared_id,   recv_count,
                    warmup_iters, profile_iters, &all_correct};
    threads.emplace_back(WorkerThread, args);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  if (!all_correct.load(std::memory_order_relaxed)) {
    std::cerr << "ReduceScatter validation failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "ReduceScatter validation passed." << std::endl;
  return EXIT_SUCCESS;
}
