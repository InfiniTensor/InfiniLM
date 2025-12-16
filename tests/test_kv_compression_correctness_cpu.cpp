#include "../src/cache_manager/kv_compression.hpp"
#include "../src/cache_manager/opcache_manager.hpp"
#include "../src/models/inference_context.hpp"
#include "../include/infinicore_infer/cache.h"

#include <infiniop.h>
#include <infinirt.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>
 

namespace {
struct Meta {
    uint32_t layers = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 0;
    uint32_t seq_in = 0;
    uint32_t seq_out = 0;
    uint32_t compression_factor = 1;
    uint32_t min_seq_len = 1;
    uint32_t image_kv_len = 0;
};

int64_t extract_int(const std::string &s, const std::string &key, int64_t def = -1) {
    std::regex re(key + R"(\s*:\s*([0-9]+))");
    std::smatch m;
    if (std::regex_search(s, m, re)) {
        return std::stoll(m[1]);
    }
    return def;
}

Meta parse_meta(const std::string &path) {
    std::ifstream fin(path);
    std::string content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    Meta m;
    m.layers = extract_int(content, "\"layers\"");
    m.heads = extract_int(content, "\"heads\"");
    m.head_dim = extract_int(content, "\"head_dim\"");
    m.seq_in = extract_int(content, "\"seq_len_in\"");
    m.seq_out = extract_int(content, "\"seq_len_out\"");
    m.compression_factor = extract_int(content, "\"compression_factor\"", 1);
    m.min_seq_len = extract_int(content, "\"min_seq_len\"", 1);
    // it_len: [a, b]
    std::regex it_re(R"("it_len"\s*:\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\])");
    std::smatch mm;
    if (std::regex_search(content, mm, it_re)) {
        m.image_kv_len = static_cast<uint32_t>(std::stoul(mm[1]));
    }
    return m;
}

std::vector<uint16_t> load_bin(const std::string &path) {
    std::ifstream fin(path, std::ios::binary);
    fin.seekg(0, std::ios::end);
    size_t bytes = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<uint16_t> buf(bytes / sizeof(uint16_t));
    fin.read(reinterpret_cast<char *>(buf.data()), bytes);
    return buf;
}

std::vector<float> tensor_to_float(std::shared_ptr<Tensor> t) {
    size_t n = t->numel();
    std::vector<float> out(n);
    std::vector<uint16_t> tmp(n);
    RUN_INFINI(infinirtMemcpy(tmp.data(), t->data(), n * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
    for (size_t i = 0; i < n; ++i) out[i] = f16_to_f32(tmp[i]);
    return out;
}

void copy_from_bin(std::shared_ptr<Tensor> t, const std::vector<uint16_t> &buf, size_t offset_elems, uint32_t heads, uint32_t seq, uint32_t dk) {
    // Source layout: [B=1, H, S, D] row-major in buf; target Tensor layout: [S, H, D]
    std::vector<uint16_t> tmp(t->numel());
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t s = 0; s < seq; ++s) {
            for (uint32_t d = 0; d < dk; ++d) {
                size_t src_idx = offset_elems + ((h * seq + s) * dk + d);
                size_t dst_idx = (static_cast<size_t>(s) * heads + h) * dk + d;
                tmp[dst_idx] = buf[src_idx];
            }
        }
    }
    RUN_INFINI(infinirtMemcpy(t->data(), tmp.data(), tmp.size() * sizeof(uint16_t), INFINIRT_MEMCPY_H2D));
}

std::pair<float, float> diff_stats(const std::vector<float> &a, const std::vector<float> &b) {
    float maxd = 0.f, meand = 0.f;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        float d = std::abs(a[i] - b[i]);
        maxd = std::max(maxd, d);
        meand += d;
    }
    meand /= static_cast<float>(n);
    return {maxd, meand};
}

void print_first(const std::vector<float> &v, size_t n, const std::string &label) {
    std::cout << label << " first " << n << ": ";
    for (size_t i = 0; i < std::min(n, v.size()); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n";
}




} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <weights.bin>" << std::endl;
        return 1;
    }

    Meta meta = parse_meta("dump_kv/meta.json");
    if (meta.layers == 0) {
        std::cerr << "Failed to parse dump_kv/meta.json" << std::endl;
        return 2;
    }

    CompressionConfig cfg;
    cfg.enable = true;
    cfg.weight_path = argv[1];
    cfg.image_kv_len = meta.image_kv_len;
    cfg.compression_factor = meta.compression_factor;
    cfg.min_seq_len = meta.min_seq_len;

    Compressor compressor(cfg);
    if (!compressor.loadWeights()) {
        std::cerr << "loadWeights failed" << std::endl;
        return 3;
    }

    RUN_INFINI(infinirtInit());
    RUN_INFINI(infinirtSetDevice(INFINI_DEVICE_CPU, 0));

    int dev_id = 0;
    auto kv = createKVCache(meta.layers, meta.seq_in, meta.heads, meta.head_dim, meta.head_dim,
                            INFINI_DTYPE_F16, INFINI_DEVICE_CPU, &dev_id, 1);

    auto input_buf = load_bin("dump_kv/input_kv.bin");
    auto output_buf = load_bin("dump_kv/output_kv.bin");

    for(int i = 0; i < 10; ++i) {
        std::cout << f16_to_f32(input_buf[i]) << std::endl;
    }
    // Load input K/V per layer using offsets from meta.index (deterministic order K then V).
    size_t elems_per = static_cast<size_t>(meta.heads) * meta.seq_in * meta.head_dim;
    for (size_t layer = 0; layer < meta.layers; ++layer) {
        size_t k_off = layer * 2 * elems_per;
        size_t v_off = k_off + elems_per;
        copy_from_bin(kv->k[0][layer], input_buf, k_off, meta.heads, meta.seq_in, meta.head_dim);
        copy_from_bin(kv->v[0][layer], input_buf, v_off, meta.heads, meta.seq_in, meta.head_dim);
    }

    auto pool = std::make_shared<MemoryPool>(256 * 1024 * 1024);
    CacheManager cache_mgr(32);
    infiniopHandle_t handle = nullptr;
    infinirtStream_t stream = nullptr;
    RUN_INFINI(infiniopCreateHandle(&handle));
    RUN_INFINI(infinirtStreamCreate(&stream));
    InferenceContext ctx(handle, pool, &cache_mgr, stream);
    setInferenceContext(&ctx);

    auto compressed = compressor.compress(*kv, meta.seq_in);
    if (!compressed) {
        std::cerr << "compress returned nullptr" << std::endl;
        return 4;
    }

    float max_diff = 0.f, mean_accum = 0.f;
    size_t total_elems = 0;
    size_t elems_out_per = static_cast<size_t>(meta.heads) * meta.seq_out * meta.head_dim;

    for (size_t layer = 0; layer < meta.layers; ++layer) {
        auto &layer_out = compressed->layers[layer];
        auto k_vec = tensor_to_float(layer_out.k_comp);
        auto v_vec = tensor_to_float(layer_out.v_comp);
        size_t k_off = layer * 2 * elems_out_per;
        size_t v_off = k_off + elems_out_per;

        std::vector<float> k_exp(elems_out_per), v_exp(elems_out_per);
        for (size_t i = 0; i < elems_out_per; ++i) {
            k_exp[i] = f16_to_f32(output_buf[k_off + i]);
            v_exp[i] = f16_to_f32(output_buf[v_off + i]);
        }
        if (layer == 0) {
            print_first(k_vec, 8, "k_out");
            print_first(k_exp, 8, "k_exp");
            print_first(v_vec, 8, "v_out");
            print_first(v_exp, 8, "v_exp");
        }
        auto kd = diff_stats(k_vec, k_exp);
        auto vd = diff_stats(v_vec, v_exp);
        max_diff = std::max({max_diff, kd.first, vd.first});
        mean_accum += (kd.second * k_vec.size() + vd.second * v_vec.size());
        total_elems += k_vec.size() + v_vec.size();
    }

    float mean_diff = mean_accum / static_cast<float>(total_elems);
    std::cout << "max_diff=" << max_diff << " mean_diff=" << mean_diff << std::endl;

    dropKVCache(kv);
    RUN_INFINI(infinirtStreamDestroy(stream));
    RUN_INFINI(infiniopDestroyHandle(handle));
    return (max_diff < 1e-3f) ? 0 : 5;
}
