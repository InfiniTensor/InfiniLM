#include "marlin_utils.hpp"
#include "marlin_support.hpp"

#include "infinicore/context/context.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

#if INFINILM_ENABLE_MARLIN
#include <infiniop/ops/awq_marlin_repack.h>
#include <infiniop/ops/gptq_marlin_repack.h>
#endif

namespace infinilm::quantization::marlin {

namespace {

#if INFINILM_ENABLE_MARLIN
void check_infiniop_status(infiniStatus_t status, const char *expr) {
    if (status != INFINI_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(expr) + " failed with status " + std::to_string(static_cast<int>(status)));
    }
}

template <typename Desc, auto Destroy>
class DescriptorGuard {
public:
    explicit DescriptorGuard(Desc desc) : desc_(desc) {}
    DescriptorGuard(const DescriptorGuard &) = delete;
    DescriptorGuard &operator=(const DescriptorGuard &) = delete;
    ~DescriptorGuard() {
        if (desc_ != nullptr) {
            Destroy(desc_);
        }
    }
    Desc get() const {
        return desc_;
    }

private:
    Desc desc_;
};

infinicore::Tensor make_workspace(size_t workspace_size, const infinicore::Device &device) {
    if (workspace_size == 0) {
        return infinicore::Tensor();
    }
    return infinicore::Tensor::empty({workspace_size}, infinicore::DataType::U8, device);
}

void *workspace_data(infinicore::Tensor &workspace) {
    return workspace ? workspace->data() : nullptr;
}

infinicore::Tensor awq_marlin_repack_gpu(const infinicore::Tensor &qweight, size_t size_k, size_t size_n, int num_bits) {
    const size_t pack_factor = 32 / num_bits;
    auto qweight_contiguous = qweight->is_contiguous() ? qweight : qweight->contiguous();
    auto output = infinicore::Tensor::empty({size_k / 16, size_n * 16 / pack_factor}, infinicore::DataType::I32, qweight_contiguous->device());

    infiniopAwqMarlinRepackDescriptor_t raw_desc = nullptr;
    check_infiniop_status(
        infiniopCreateAwqMarlinRepackDescriptor(
            infinicore::context::getInfiniopHandle(qweight_contiguous->device()),
            &raw_desc,
            output->desc(),
            qweight_contiguous->desc(),
            num_bits,
            false),
        "infiniopCreateAwqMarlinRepackDescriptor");
    DescriptorGuard<infiniopAwqMarlinRepackDescriptor_t, infiniopDestroyAwqMarlinRepackDescriptor> desc(raw_desc);

    size_t workspace_size = 0;
    check_infiniop_status(
        infiniopGetAwqMarlinRepackWorkspaceSize(desc.get(), &workspace_size),
        "infiniopGetAwqMarlinRepackWorkspaceSize");
    auto workspace = make_workspace(workspace_size, qweight_contiguous->device());

    check_infiniop_status(
        infiniopAwqMarlinRepack(
            desc.get(),
            workspace_data(workspace),
            workspace_size,
            output->data(),
            qweight_contiguous->data(),
            infinicore::context::getStream()),
        "infiniopAwqMarlinRepack");
    infinicore::context::syncStream();
    return output;
}

infinicore::Tensor gptq_marlin_repack_gpu(
    const infinicore::Tensor &qweight,
    const infinicore::Tensor &perm,
    size_t size_k,
    size_t size_n,
    int num_bits) {
    const size_t pack_factor = 32 / num_bits;
    auto qweight_contiguous = qweight->is_contiguous() ? qweight : qweight->contiguous();
    auto output = infinicore::Tensor::empty({size_k / 16, size_n * 16 / pack_factor}, infinicore::DataType::I32, qweight_contiguous->device());
    auto perm_desc = (perm && perm->numel() != 0) ? perm->desc() : nullptr;
    const void *perm_data = (perm && perm->numel() != 0) ? perm->data() : nullptr;

    infiniopGptqMarlinRepackDescriptor_t raw_desc = nullptr;
    check_infiniop_status(
        infiniopCreateGptqMarlinRepackDescriptor(
            infinicore::context::getInfiniopHandle(qweight_contiguous->device()),
            &raw_desc,
            output->desc(),
            qweight_contiguous->desc(),
            perm_desc,
            num_bits,
            false),
        "infiniopCreateGptqMarlinRepackDescriptor");
    DescriptorGuard<infiniopGptqMarlinRepackDescriptor_t, infiniopDestroyGptqMarlinRepackDescriptor> desc(raw_desc);

    size_t workspace_size = 0;
    check_infiniop_status(
        infiniopGetGptqMarlinRepackWorkspaceSize(desc.get(), &workspace_size),
        "infiniopGetGptqMarlinRepackWorkspaceSize");
    auto workspace = make_workspace(workspace_size, qweight_contiguous->device());

    check_infiniop_status(
        infiniopGptqMarlinRepack(
            desc.get(),
            workspace_data(workspace),
            workspace_size,
            output->data(),
            qweight_contiguous->data(),
            perm_data,
            infinicore::context::getStream()),
        "infiniopGptqMarlinRepack");
    infinicore::context::syncStream();
    return output;
}
#endif

std::vector<int> scale_perm() {
    std::vector<int> perm;
    perm.reserve(64);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            perm.push_back(i + 8 * j);
        }
    }
    return perm;
}

std::vector<int> scale_perm_single() {
    std::vector<int> perm;
    perm.reserve(32);
    for (int i = 0; i < 4; ++i) {
        for (int j : {0, 1, 8, 9, 16, 17, 24, 25}) {
            perm.push_back(2 * i + j);
        }
    }
    return perm;
}

infinicore::Tensor to_cpu_contiguous(const infinicore::Tensor &tensor) {
    return tensor->contiguous()->to(infinicore::Device::cpu());
}

infinicore::Tensor copy_to_device(const void *data, size_t bytes, const std::vector<size_t> &shape,
                                  infinicore::DataType dtype, const infinicore::Device &device) {
    auto cpu = infinicore::Tensor::empty(shape, dtype, infinicore::Device::cpu());
    if (bytes != 0) {
        std::memcpy(cpu->data(), data, bytes);
    }
    return device == infinicore::Device::cpu() ? cpu : cpu->to(device);
}

std::vector<int32_t> unpack_cols(const int32_t *packed, size_t size_k, size_t size_n, int num_bits) {
    const size_t pack_factor = 32 / num_bits;
    std::vector<int32_t> out(size_k * size_n, 0);
    const uint32_t mask = (1u << num_bits) - 1u;
    for (size_t r = 0; r < size_k; ++r) {
        for (size_t c_pack = 0; c_pack < size_n / pack_factor; ++c_pack) {
            uint32_t word = static_cast<uint32_t>(packed[r * (size_n / pack_factor) + c_pack]);
            for (size_t i = 0; i < pack_factor; ++i) {
                out[r * size_n + c_pack * pack_factor + i] = static_cast<int32_t>(word & mask);
                word >>= num_bits;
            }
        }
    }
    return out;
}

std::vector<int32_t> pack_cols(const std::vector<int32_t> &unpacked, size_t size_k, size_t size_n, int num_bits) {
    const size_t pack_factor = 32 / num_bits;
    std::vector<int32_t> out(size_k * (size_n / pack_factor), 0);
    for (size_t r = 0; r < size_k; ++r) {
        for (size_t c_pack = 0; c_pack < size_n / pack_factor; ++c_pack) {
            uint32_t word = 0;
            for (size_t i = 0; i < pack_factor; ++i) {
                uint32_t value = static_cast<uint32_t>(unpacked[r * size_n + c_pack * pack_factor + i]);
                word |= value << (num_bits * i);
            }
            out[r * (size_n / pack_factor) + c_pack] = static_cast<int32_t>(word);
        }
    }
    return out;
}

void check_repack_shape(size_t size_k, size_t size_n, int num_bits) {
    if (num_bits != 4 && num_bits != 8) {
        throw std::runtime_error("marlin repack: num_bits must be 4 or 8");
    }
    if (size_k % 16 != 0 || size_n % 64 != 0) {
        throw std::runtime_error("marlin repack: size_k must be divisible by 16 and size_n by 64");
    }
}

uint32_t pack_repack_values(const uint32_t *vals, int num_bits, bool upper_half) {
    if (num_bits == 4) {
        constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        uint32_t word = 0;
        for (int i = 0; i < 8; ++i) {
            word |= vals[pack_idx[i]] << (i * 4);
        }
        return word;
    }

    constexpr int pack_idx[4] = {0, 2, 1, 3};
    uint32_t word = 0;
    const int offset = upper_half ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        word |= vals[offset + pack_idx[i]] << (i * 8);
    }
    return word;
}

uint32_t awq_value(const int32_t *packed, size_t size_n, size_t k, size_t n, int num_bits) {
    const size_t pack_factor = 32 / num_bits;
    const uint32_t mask = (1u << num_bits) - 1u;
    const size_t n_pack = n / pack_factor;
    const size_t n_pos = n % pack_factor;
    constexpr int undo4[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    constexpr int undo8[4] = {0, 2, 1, 3};
    const int pos = num_bits == 4 ? undo4[n_pos] : undo8[n_pos];
    const auto word = static_cast<uint32_t>(packed[k * (size_n / pack_factor) + n_pack]);
    return (word >> (pos * num_bits)) & mask;
}

uint32_t gptq_value(const int32_t *packed, size_t size_n, size_t k, size_t n, int num_bits) {
    const size_t pack_factor = 32 / num_bits;
    const uint32_t mask = (1u << num_bits) - 1u;
    const size_t k_pack = k / pack_factor;
    const size_t k_pos = k % pack_factor;
    const auto word = static_cast<uint32_t>(packed[k_pack * size_n + n]);
    return (word >> (k_pos * num_bits)) & mask;
}

template <typename Getter>
std::vector<int32_t> repack_to_marlin_tiles(size_t size_k, size_t size_n, int num_bits, Getter get_value) {
    constexpr size_t tile_k = 16;
    constexpr size_t tile_n = 64;
    const size_t pack_factor = 32 / num_bits;
    const size_t k_tiles = size_k / tile_k;
    const size_t n_tiles = size_n / tile_n;
    const size_t tile_words = tile_k * tile_n / pack_factor;
    std::vector<int32_t> out(k_tiles * n_tiles * tile_words, 0);
    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    for (size_t kt = 0; kt < k_tiles; ++kt) {
        for (size_t nt = 0; nt < n_tiles; ++nt) {
            const size_t out_offset = (kt * n_tiles + nt) * tile_words;
            for (int warp = 0; warp < 4; ++warp) {
                for (int th = 0; th < 32; ++th) {
                    const int tc_col = th / 4;
                    const int tc_row = (th % 4) * 2;
                    const size_t n0 = nt * tile_n + warp * 16 + tc_col;

                    uint32_t vals[8];
                    for (int i = 0; i < 4; ++i) {
                        const size_t k = kt * tile_k + static_cast<size_t>(tc_row + tc_offsets[i]);
                        vals[i] = get_value(k, n0);
                        vals[4 + i] = get_value(k, n0 + 8);
                    }

                    if (num_bits == 4) {
                        out[out_offset + static_cast<size_t>(th * 4 + warp)] = static_cast<int32_t>(pack_repack_values(vals, num_bits, false));
                    } else {
                        out[out_offset + static_cast<size_t>(th * 8 + warp * 2)] = static_cast<int32_t>(pack_repack_values(vals, num_bits, false));
                        out[out_offset + static_cast<size_t>(th * 8 + warp * 2 + 1)] = static_cast<int32_t>(pack_repack_values(vals, num_bits, true));
                    }
                }
            }
        }
    }
    return out;
}

} // namespace

bool supports_shape(size_t input_size_per_partition, size_t output_size_per_partition, int group_size) {
    if (output_size_per_partition % 64 != 0 || input_size_per_partition % 128 != 0) {
        return false;
    }
    if (!(group_size == -1 || group_size == 32 || group_size == 64 || group_size == 128)) {
        return false;
    }
    return group_size == -1 || input_size_per_partition % static_cast<size_t>(group_size) == 0;
}

infinicore::Tensor make_empty_i32(const infinicore::Device &device) {
    return infinicore::Tensor::empty({0, 0}, infinicore::DataType::I32, device);
}

infinicore::Tensor make_i32_tensor(const std::vector<int32_t> &data, const std::vector<size_t> &shape, const infinicore::Device &device) {
    return copy_to_device(data.data(), data.size() * sizeof(int32_t), shape, infinicore::DataType::I32, device);
}

infinicore::Tensor awq_marlin_repack(const infinicore::Tensor &qweight, size_t size_k, size_t size_n, int num_bits) {
    check_repack_shape(size_k, size_n, num_bits);
    const size_t pack_factor = 32 / num_bits;
#if INFINILM_ENABLE_MARLIN
    if (qweight->dtype() != infinicore::DataType::I32 || qweight->shape() != std::vector<size_t>{size_k, size_n / pack_factor}) {
        throw std::runtime_error("awq_marlin_repack: unexpected qweight shape or dtype");
    }
    if (qweight->device().getType() == infinicore::Device::Type::NVIDIA) {
        return awq_marlin_repack_gpu(qweight, size_k, size_n, num_bits);
    }
#endif
    auto cpu = to_cpu_contiguous(qweight);
    if (cpu->dtype() != infinicore::DataType::I32 || cpu->shape() != std::vector<size_t>{size_k, size_n / pack_factor}) {
        throw std::runtime_error("awq_marlin_repack: unexpected qweight shape or dtype");
    }
    auto *packed = reinterpret_cast<const int32_t *>(cpu->data());
    auto out = repack_to_marlin_tiles(size_k, size_n, num_bits, [&](size_t k, size_t n) {
        return awq_value(packed, size_n, k, n, num_bits);
    });
    return make_i32_tensor(out, {size_k / 16, size_n * 16 / pack_factor}, qweight->device());
}

infinicore::Tensor gptq_marlin_repack(const infinicore::Tensor &qweight, const infinicore::Tensor &perm, size_t size_k, size_t size_n, int num_bits) {
    check_repack_shape(size_k, size_n, num_bits);
    const size_t pack_factor = 32 / num_bits;
#if INFINILM_ENABLE_MARLIN
    if (qweight->dtype() != infinicore::DataType::I32 || qweight->shape() != std::vector<size_t>{size_k / pack_factor, size_n}) {
        throw std::runtime_error("gptq_marlin_repack: unexpected qweight shape or dtype");
    }
    if (perm && perm->numel() != 0 && (perm->dtype() != infinicore::DataType::I32 || perm->numel() != size_k)) {
        throw std::runtime_error("gptq_marlin_repack: unexpected perm shape or dtype");
    }
    if (qweight->device().getType() == infinicore::Device::Type::NVIDIA) {
        return gptq_marlin_repack_gpu(qweight, perm, size_k, size_n, num_bits);
    }
#endif
    auto cpu = to_cpu_contiguous(qweight);
    if (cpu->dtype() != infinicore::DataType::I32 || cpu->shape() != std::vector<size_t>{size_k / pack_factor, size_n}) {
        throw std::runtime_error("gptq_marlin_repack: unexpected qweight shape or dtype");
    }
    auto *packed = reinterpret_cast<const int32_t *>(cpu->data());

    std::vector<int32_t> perm_data;
    if (perm && perm->numel() != 0) {
        auto perm_cpu = to_cpu_contiguous(perm);
        if (perm_cpu->dtype() != infinicore::DataType::I32 || perm_cpu->numel() != size_k) {
            throw std::runtime_error("gptq_marlin_repack: unexpected perm shape or dtype");
        }
        auto *src = reinterpret_cast<const int32_t *>(perm_cpu->data());
        perm_data.assign(src, src + perm_cpu->numel());
    }

    auto out = repack_to_marlin_tiles(size_k, size_n, num_bits, [&](size_t k, size_t n) {
        const size_t src_k = perm_data.empty() ? k : static_cast<size_t>(perm_data[k]);
        return gptq_value(packed, size_n, src_k, n, num_bits);
    });
    return make_i32_tensor(out, {size_k / 16, size_n * 16 / pack_factor}, qweight->device());
}

infinicore::Tensor sort_g_idx(const infinicore::Tensor &g_idx, infinicore::Tensor &sort_indices) {
    auto cpu = to_cpu_contiguous(g_idx);
    const auto size = cpu->numel();
    auto *gidx_data = reinterpret_cast<const int32_t *>(cpu->data());

    std::vector<int32_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [gidx_data](int32_t lhs, int32_t rhs) {
        return gidx_data[lhs] < gidx_data[rhs];
    });

    std::vector<int32_t> sorted(size);
    for (size_t i = 0; i < size; ++i) {
        sorted[i] = gidx_data[indices[i]];
    }

    sort_indices = make_i32_tensor(indices, {size}, g_idx->device());
    return make_i32_tensor(sorted, {size}, g_idx->device());
}

infinicore::Tensor permute_scales(const infinicore::Tensor &scales, size_t size_k, size_t size_n, int group_size) {
    auto cpu = to_cpu_contiguous(scales);
    const auto elem_size = cpu->element_size();
    const auto *src = reinterpret_cast<const std::byte *>(cpu->data());
    std::vector<std::byte> dst(cpu->nbytes());

    const auto perm = (group_size < static_cast<int>(size_k) && group_size != -1) ? scale_perm() : scale_perm_single();
    const size_t block = perm.size();
    if (cpu->numel() % block != 0) {
        throw std::runtime_error("marlin permute_scales: scale tensor size is not compatible with Marlin permutation");
    }
    for (size_t row = 0; row < cpu->numel() / block; ++row) {
        for (size_t i = 0; i < block; ++i) {
            std::memcpy(
                dst.data() + (row * block + i) * elem_size,
                src + (row * block + static_cast<size_t>(perm[i])) * elem_size,
                elem_size);
        }
    }
    return copy_to_device(dst.data(), dst.size(), {cpu->size(0), size_n}, cpu->dtype(), scales->device());
}

infinicore::Tensor awq_to_marlin_zero_points(const infinicore::Tensor &qzeros, size_t size_k, size_t size_n, int num_bits) {
    auto cpu = to_cpu_contiguous(qzeros);
    const size_t pack_factor = 32 / num_bits;
    if (cpu->shape() != std::vector<size_t>{size_k, size_n / pack_factor}) {
        throw std::runtime_error("awq_to_marlin_zero_points: unexpected qzeros shape");
    }

    auto unpacked = unpack_cols(reinterpret_cast<const int32_t *>(cpu->data()), size_k, size_n, num_bits);
    const std::vector<int> undo_interleave = num_bits == 4 ? std::vector<int>{0, 4, 1, 5, 2, 6, 3, 7}
                                                           : std::vector<int>{0, 2, 1, 3};
    std::vector<int32_t> unpermuted(unpacked.size());
    for (size_t row = 0; row < unpacked.size() / undo_interleave.size(); ++row) {
        for (size_t i = 0; i < undo_interleave.size(); ++i) {
            unpermuted[row * undo_interleave.size() + i] = unpacked[row * undo_interleave.size() + static_cast<size_t>(undo_interleave[i])];
        }
    }

    auto perm = scale_perm();
    if (unpermuted.size() % perm.size() != 0) {
        throw std::runtime_error("awq_to_marlin_zero_points: zero-point tensor size is not compatible with Marlin permutation");
    }
    std::vector<int32_t> permuted(unpermuted.size());
    for (size_t row = 0; row < unpermuted.size() / perm.size(); ++row) {
        for (size_t i = 0; i < perm.size(); ++i) {
            permuted[row * perm.size() + i] = unpermuted[row * perm.size() + static_cast<size_t>(perm[i])];
        }
    }

    const std::vector<int> interleave = num_bits == 4 ? std::vector<int>{0, 2, 4, 6, 1, 3, 5, 7}
                                                      : std::vector<int>{0, 2, 1, 3};
    std::vector<int32_t> interleaved(permuted.size());
    for (size_t row = 0; row < permuted.size() / interleave.size(); ++row) {
        for (size_t i = 0; i < interleave.size(); ++i) {
            interleaved[row * interleave.size() + i] = permuted[row * interleave.size() + static_cast<size_t>(interleave[i])];
        }
    }

    auto packed = pack_cols(interleaved, size_k, size_n, num_bits);
    return make_i32_tensor(packed, {size_k, size_n / pack_factor}, qzeros->device());
}

infinicore::Tensor permute_bias(const infinicore::Tensor &bias) {
    auto cpu = to_cpu_contiguous(bias);
    const auto elem_size = cpu->element_size();
    const auto *src = reinterpret_cast<const std::byte *>(cpu->data());
    std::vector<std::byte> dst(cpu->nbytes());
    const auto perm = scale_perm_single();
    if (cpu->numel() % perm.size() != 0) {
        throw std::runtime_error("marlin permute_bias: bias tensor size is not compatible with Marlin permutation");
    }
    for (size_t row = 0; row < cpu->numel() / perm.size(); ++row) {
        for (size_t i = 0; i < perm.size(); ++i) {
            std::memcpy(
                dst.data() + (row * perm.size() + i) * elem_size,
                src + (row * perm.size() + static_cast<size_t>(perm[i])) * elem_size,
                elem_size);
        }
    }
    return copy_to_device(dst.data(), dst.size(), cpu->shape(), cpu->dtype(), bias->device());
}

} // namespace infinilm::quantization::marlin
