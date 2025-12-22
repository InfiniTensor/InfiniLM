#include "../src/cache_manager/kv_compression.hpp"
#include "../src/models/inference_context.hpp" // to init context for linear/relu
#include "../src/cache_manager/opcache_manager.hpp"
#include "../include/infinicore_infer/cache.h"
#include <infinirt.h>
#include <infiniop.h>

#include <iostream>
#include <memory>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <weights.bin> [cpu|hygon]" << std::endl;
        return 1;
    }
    std::string dev_arg = argc >= 3 ? std::string(argv[2]) : "cpu";

    CompressionConfig cfg;
    cfg.enable = true;
    cfg.weight_path = argv[1];

    // Prepare a fake KVCache (single device, single layer).
    size_t nlayers = 1;
    size_t max_len = 20;
    size_t nkvh = 32; // match LLAVA config (num_key_value_heads)
    size_t dk = 128;  // head_dim; cols = head_dim * factor (128*5=640)
    size_t dv = 128;
    // Use Hygon device as requested
    infiniDevice_t device = INFINI_DEVICE_CPU;
    if (dev_arg == "hygon") {
        device = INFINI_DEVICE_HYGON;
    }
    int dev_id = 0;

    RUN_INFINI(infinirtInit());
    RUN_INFINI(infinirtSetDevice(device, dev_id));

    Compressor compressor(cfg);
    if (!compressor.loadWeights()) {
        std::cerr << "loadWeights failed" << std::endl;
        return 2;
    }

    auto kv = createKVCache(nlayers, max_len, nkvh, dk, dv, INFINI_DTYPE_F16, device, &dev_id, 1);

    // Initialize a minimal inference context (device-specific) for linear/relu ops.
    auto pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);
    CacheManager cache_mgr(32);
    infiniopHandle_t handle = nullptr;
    infinirtStream_t stream = nullptr;
    RUN_INFINI(infiniopCreateHandle(&handle));
    RUN_INFINI(infinirtStreamCreate(&stream));
    InferenceContext ctx(handle, pool, &cache_mgr, stream);
    setInferenceContext(&ctx);

    auto compressed = compressor.compress(*kv, static_cast<uint32_t>(max_len));
    if (!compressed) {
        std::cerr << "compress returned nullptr (likely missing op_handle/memory_pool)" << std::endl;
    } else {
        auto &layer0 = compressed->layers[0];
        std::cout << "Compressed seq_len=" << layer0.comp_seq_len
                  << " orig=" << layer0.orig_seq_len << std::endl;
        if (layer0.k_comp) {
            std::cout << "k_comp shape: ";
            for (auto d : layer0.k_comp->shape()) std::cout << d << " ";
            std::cout << "\n";
        }
        if (layer0.v_comp) {
            std::cout << "v_comp shape: ";
            for (auto d : layer0.v_comp->shape()) std::cout << d << " ";
            std::cout << "\n";
        }
    }

    dropKVCache(kv);
    RUN_INFINI(infinirtStreamDestroy(stream));
    RUN_INFINI(infiniopDestroyHandle(handle));
    return 0;
}
