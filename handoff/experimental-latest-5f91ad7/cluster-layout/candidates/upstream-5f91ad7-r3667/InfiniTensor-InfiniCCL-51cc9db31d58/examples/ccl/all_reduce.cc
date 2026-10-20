/**
 * InfiniCCL Example: Thread-per-GPU Single-Node AllReduce
 *
 * This example demonstrates spawning one CPU thread per GPU on a single node,
 * generating a shared UniqueID, and performing an AllReduce entirely via
 * InfiniCCL's native CCL backend without any OpenMPI dependencies.
 *
 * Note: to properly run this example, you should specify `--launcher none`
 * since it requires no MPI process spawning.
 */

#include <unistd.h>

#include <iostream>
#include <numeric>
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

// Structure to pass execution data to each GPU worker thread.
struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iter;
  int profile_iter;
};

// Worker function executed by each CPU thread.
void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  // Bind this specific CPU thread to its designated local GPU device
  // In a thread-per-GPU model, `local_rank` is exactly the thread's rank ID.
  int local_device_id = args.rank;
  CHECK_RT(Rt, Rt::SetDevice(local_device_id));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  // Prepare Host Data Structures
  std::vector<float> h_send(args.num_elements);
  std::vector<float> h_recv(args.num_elements, 0.0f);

  // Each rank provides its own (rank + 1) as data.
  for (size_t i = 0; i < args.num_elements; i++) {
    h_send[i] = static_cast<float>(args.rank + 1);
  }

  // Allocate GPU Memory using InfiniCCL's Runtime abstraction layer.
  float *d_send = nullptr;
  float *d_recv = nullptr;
  size_t total_bytes = args.num_elements * sizeof(float);

  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, total_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, total_bytes));

  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up Iterations
  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(infinicclAllReduce(d_send, d_recv, args.num_elements,
                                    infinicclFloat32, infinicclSum, comm,
                                    nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling Iterations
  Timer timer;
  for (int i = 0; i < args.profile_iter; i++) {
    CHECK_INFINI(infinicclAllReduce(d_send, d_recv, args.num_elements,
                                    infinicclFloat32, infinicclSum, comm,
                                    nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  double elapsed = timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  // Copy output reduced data back from device to host.
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                          Rt::MemcpyDeviceToHost));

  // Result Validation
  float expected = 0.0f;
  for (int r = 0; r < args.size; r++) {
    expected += static_cast<float>(r + 1);
  }
  Validator::ValidateResult(h_recv.data(), args.num_elements, expected,
                            args.rank, true, "AllReduce");

  // Metrics Reporting (Only Rank 0)
  if (args.rank == 0) {
    std::cout << "\n=== Single-Node Threaded AllReduce Results ==="
              << std::endl;
    std::cout << "Data size: " << args.num_elements << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    Metrics metrics{elapsed, total_bytes, args.size};
    metrics.Print();
  }

  // Cleanup local rank resources.
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
}

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iters = 1;
  int profile_iters = 20;
  size_t num_elements = 1 << 25;

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
                  << "  -w <warmup_iters>    Warmup iterations (default: 1)\n"
                  << "  -p <profile_iters>   Profile iterations (default: 20)\n"
                  << "  -n <num_elements>    Number of elements (default: "
                  << (1 << 25) << ")\n";
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

  // Spawn CPU thread pool.
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);

  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,         num_gpus,     shared_id,
                    num_elements, warmup_iters, profile_iters};
    threads.emplace_back(WorkerThread, args);
  }

  // Await execution completion across all threads.
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  std::cout
      << "[Main Process] All worker threads joined. InfiniCCL finalized safely."
      << std::endl;
  return EXIT_SUCCESS;
}
