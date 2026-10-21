#include "infinicore/ops/rope.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/nn/rope.hpp"

#include <rope_kernel.h>
#include <rope_pack_kernel.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

namespace infinicore::op::rope_impl::aten {
namespace {

// QY FA+paged long-prefill RoPE: Denglin ATen index_select/gather and strided
// copy_/mul on gated as_strided Q/K hit LaunchOutOfResources. Pack/unpack with
// cudaMemcpyAsync (contiguous head vectors), rotate on a compact workspace,
// gather cos/sin via cache-table slice.
constexpr int64_t kQyRopeChunkTokens = 128;



struct PlannedMeta {
    Tensor x_out;
    Tensor x;
    Tensor pos;
    Tensor sin;
    Tensor cos;
    infinicore::nn::RoPE::Algo algo;
};

void *plan(Tensor x_out,
           const Tensor &x,
           const Tensor &pos,
           const Tensor &sin,
           const Tensor &cos,
           infinicore::nn::RoPE::Algo algo) {
    INFINICORE_ASSERT(x_out->device().type() == Device::Type::kQy);
    return new PlannedMeta{x_out, x, pos, sin, cos, algo};
}

int64_t seq_dim_of(const at::Tensor &x) {
    if (x.dim() == 2) {
        return 0;
    }
    INFINICORE_ASSERT(x.dim() >= 3);
    return x.dim() - 3;
}

at::Tensor d2h_contig(const at::Tensor &dev) {
    INFINICORE_ASSERT(dev.is_contiguous());
    auto host = at::empty(dev.sizes(), dev.options().device(at::kCPU));
    cudaMemcpy(host.data_ptr(),
               dev.data_ptr(),
               static_cast<size_t>(dev.numel() * static_cast<int64_t>(dev.element_size())),
               cudaMemcpyDeviceToHost);
    return host;
}

at::Tensor gather_cache_rows(const at::Tensor &table, const at::Tensor &pos_chunk) {
    const int64_t rows = pos_chunk.numel();
    INFINICORE_ASSERT(pos_chunk.dim() == 1);
    INFINICORE_ASSERT(table.dim() == 2);
    if (rows == 0) {
        return at::empty({0, table.size(1)}, table.options());
    }

    // Host indices once — avoid ATen index_select / .contiguous() on device.
    INFINICORE_ASSERT(pos_chunk.is_contiguous());
    auto pos_host = d2h_contig(pos_chunk);
    const auto *pos_ptr = pos_host.data_ptr<int64_t>();

    auto out = at::empty({rows, table.size(1)}, table.options());
    const auto elem = static_cast<int64_t>(table.element_size());
    auto *dst_base = static_cast<char *>(out.data_ptr());
    const auto *src_base = static_cast<const char *>(table.data_ptr());
    const int64_t row_bytes = table.size(1) * elem;
    const int64_t table_rows = table.size(0);

    const int64_t p0 = pos_ptr[0];
    bool consecutive = p0 >= 0 && (p0 + rows) <= table_rows;
    for (int64_t i = 0; consecutive && i < rows; ++i) {
        consecutive = (pos_ptr[i] == p0 + i);
    }
    if (consecutive && table.stride(0) == table.size(1)) {
        cudaMemcpy(dst_base,
                   src_base + p0 * table.stride(0) * elem,
                   static_cast<size_t>(rows * row_bytes),
                   cudaMemcpyDeviceToDevice);
        return out;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const int64_t p = pos_ptr[i];
        INFINICORE_ASSERT(p >= 0 && p < table_rows);
        cudaMemcpy(dst_base + i * row_bytes,
                   src_base + p * table.stride(0) * elem,
                   static_cast<size_t>(row_bytes),
                   cudaMemcpyDeviceToDevice);
    }
    return out;
}

// Pack strided [seq, heads, dim] (or [seq, dim]) into contiguous via QY kernel.
void pack_head_vectors(const at::Tensor &src, at::Tensor &dst, cudaStream_t stream) {
    INFINICORE_ASSERT(src.sizes() == dst.sizes());
    INFINICORE_ASSERT(dst.is_contiguous());
    if (src.is_contiguous()) {
        cudaMemcpyAsync(dst.data_ptr(),
                        src.data_ptr(),
                        static_cast<size_t>(src.numel() * static_cast<int64_t>(src.element_size())),
                        cudaMemcpyDeviceToDevice,
                        stream);
        return;
    }
    const int elem = static_cast<int>(src.element_size());
    const int64_t dim = src.size(-1);
    INFINICORE_ASSERT(src.stride(-1) == 1);

    if (src.dim() == 2) {
        ::infini::ops::qy::rope_pack_launch(
            dst.data_ptr(), src.data_ptr(), static_cast<size_t>(src.size(0)),
            /*heads=*/1, static_cast<size_t>(dim), src.stride(0), /*s1=*/0,
            /*s2=*/1, elem, reinterpret_cast<void *>(stream));
        return;
    }
    INFINICORE_ASSERT(src.dim() == 3);
    ::infini::ops::qy::rope_pack_launch(
        dst.data_ptr(), src.data_ptr(), static_cast<size_t>(src.size(0)),
        static_cast<size_t>(src.size(1)), static_cast<size_t>(dim),
        src.stride(0), src.stride(1), src.stride(2), elem,
        reinterpret_cast<void *>(stream));
}

void unpack_head_vectors(const at::Tensor &src, at::Tensor &dst, cudaStream_t stream) {
    INFINICORE_ASSERT(src.sizes() == dst.sizes());
    INFINICORE_ASSERT(src.is_contiguous());
    if (dst.is_contiguous()) {
        cudaMemcpyAsync(dst.data_ptr(),
                        src.data_ptr(),
                        static_cast<size_t>(src.numel() * static_cast<int64_t>(src.element_size())),
                        cudaMemcpyDeviceToDevice,
                        stream);
        return;
    }
    const int elem = static_cast<int>(src.element_size());
    const int64_t dim = src.size(-1);
    INFINICORE_ASSERT(dst.stride(-1) == 1);

    if (dst.dim() == 2) {
        ::infini::ops::qy::rope_unpack_launch(
            dst.data_ptr(), src.data_ptr(), static_cast<size_t>(dst.size(0)),
            /*heads=*/1, static_cast<size_t>(dim), dst.stride(0), /*s1=*/0,
            /*s2=*/1, elem, reinterpret_cast<void *>(stream));
        return;
    }
    INFINICORE_ASSERT(dst.dim() == 3);
    ::infini::ops::qy::rope_unpack_launch(
        dst.data_ptr(), src.data_ptr(), static_cast<size_t>(dst.size(0)),
        static_cast<size_t>(dst.size(1)), static_cast<size_t>(dim),
        dst.stride(0), dst.stride(1), dst.stride(2), elem,
        reinterpret_cast<void *>(stream));
}

at::Tensor rotate_contig_gpu(const at::Tensor &x_contig,
                             const at::Tensor &pos_chunk,
                             const at::Tensor &sin_table,
                             const at::Tensor &cos_table,
                             infinicore::nn::RoPE::Algo algo,
                             int dtype_tag,
                             cudaStream_t stream) {
    INFINICORE_ASSERT(x_contig.is_contiguous());
    INFINICORE_ASSERT(pos_chunk.is_contiguous());
    INFINICORE_ASSERT(sin_table.is_contiguous());
    INFINICORE_ASSERT(cos_table.is_contiguous());

    const int64_t seqlen = x_contig.size(0);
    const int64_t nhead = (x_contig.dim() == 2) ? 1 : x_contig.size(1);
    const int64_t head_dim = x_contig.size(-1);
    INFINICORE_ASSERT(pos_chunk.numel() == seqlen);
    INFINICORE_ASSERT(sin_table.size(1) == head_dim / 2);
    INFINICORE_ASSERT(cos_table.size(1) == head_dim / 2);

    const int algo_tag = (algo == infinicore::nn::RoPE::Algo::GPT_J) ? 0 : 1;
    if (algo != infinicore::nn::RoPE::Algo::GPT_NEOX && algo != infinicore::nn::RoPE::Algo::GPT_J) {
        throw std::runtime_error("Unsupported RoPE algorithm for QY ATen path");
    }

    auto out = at::empty(x_contig.sizes(), x_contig.options());
    ::infini::ops::qy::rope_launch(
        out.data_ptr(), x_contig.data_ptr(),
        pos_chunk.data_ptr<int64_t>(), sin_table.data_ptr(),
        cos_table.data_ptr(), static_cast<size_t>(seqlen),
        static_cast<size_t>(nhead), static_cast<size_t>(head_dim), algo_tag,
        dtype_tag, reinterpret_cast<void *>(stream));
    return out;
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    using namespace ::infinicore::adaptor;
    set_aten_stream_to_infinicore();


    auto x = to_aten_tensor(p->x);
    auto pos = to_aten_tensor(p->pos).to(at::kLong).reshape({-1}).contiguous();
    auto sin_table = to_aten_tensor(p->sin);
    auto cos_table = to_aten_tensor(p->cos);
    auto out = to_aten_tensor(p->x_out);

    const int64_t seq_len = pos.numel();
    const int64_t seq_dim = seq_dim_of(x);
    INFINICORE_ASSERT(seq_dim == 0);
    INFINICORE_ASSERT(x.size(0) == seq_len);
    INFINICORE_ASSERT(out.sizes() == x.sizes());
    INFINICORE_ASSERT(x.stride(-1) == 1);
    INFINICORE_ASSERT(out.stride(-1) == 1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Prefer full-table + pos lookup in the kernel to avoid per-chunk gather.
    // Fall back to gathered rows only if tables are not contiguous.
    at::Tensor sin_use = sin_table;
    at::Tensor cos_use = cos_table;
    at::Tensor pos_use = pos;
    if (!sin_table.is_contiguous() || !cos_table.is_contiguous()) {
        sin_use = gather_cache_rows(sin_table, pos);
        cos_use = gather_cache_rows(cos_table, pos);
        // Remap positions to [0..seq) against gathered rows.
        auto pos_remap = at::empty({seq_len}, pos.options().device(at::kCPU));
        auto *ptr = pos_remap.data_ptr<int64_t>();
        for (int64_t i = 0; i < seq_len; ++i) {
            ptr[i] = i;
        }
        pos_use = at::empty({seq_len}, pos.options());
        cudaMemcpy(pos_use.data_ptr(), ptr,
                   static_cast<size_t>(seq_len * static_cast<int64_t>(sizeof(int64_t))),
                   cudaMemcpyHostToDevice);
    }

    const int algo_tag =
        (p->algo == infinicore::nn::RoPE::Algo::GPT_J) ? 0 : 1;
    if (p->algo != infinicore::nn::RoPE::Algo::GPT_NEOX &&
        p->algo != infinicore::nn::RoPE::Algo::GPT_J) {
        throw std::runtime_error("Unsupported RoPE algorithm for QY ATen path");
    }
    const int dtype_tag = static_cast<int>(p->x->dtype());
    INFINICORE_ASSERT(sin_use.is_contiguous());
    INFINICORE_ASSERT(cos_use.is_contiguous());
    INFINICORE_ASSERT(pos_use.is_contiguous());

    // Fast path: strided in-place/out RoPE (no pack/unpack). Covers MiniCPM5
    // gated Q [seq,heads,dim] strides like [8704,256,1] and contiguous K.
    // Also [1,seq,heads,dim] static views with batch=1.
    const bool same_layout = out.strides() == x.strides();
    int64_t nhead = 1;
    int64_t head_dim = x.size(-1);
    int64_t s0 = 0;
    int64_t s1 = 0;
    bool use_strided = false;
    if (same_layout && x.stride(-1) == 1) {
        if (x.dim() == 2) {
            nhead = 1;
            s0 = x.stride(0);
            s1 = 0;
            use_strided = true;
        } else if (x.dim() == 3) {
            nhead = x.size(1);
            s0 = x.stride(0);
            s1 = x.stride(1);
            use_strided = true;
        } else if (x.dim() == 4 && x.size(0) == 1) {
            nhead = x.size(2);
            s0 = x.stride(1);
            s1 = x.stride(2);
            use_strided = true;
        }
    }
    if (use_strided) {
        ::infini::ops::qy::rope_strided_launch(
            out.data_ptr(), x.data_ptr(), pos_use.data_ptr<int64_t>(),
            sin_use.data_ptr(), cos_use.data_ptr(),
            static_cast<size_t>(seq_len), static_cast<size_t>(nhead),
            static_cast<size_t>(head_dim), s0, s1, /*stride2=*/1, algo_tag,
            dtype_tag, reinterpret_cast<void *>(stream));
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("QY rope_strided launch: ") +
                                     cudaGetErrorString(err));
        }
    } else {
        // Legacy pack → contiguous rotate → unpack.
        at::Tensor x_contig =
            x.is_contiguous() ? x : at::empty(x.sizes(), x.options());
        if (!x.is_contiguous()) {
            pack_head_vectors(x, x_contig, stream);
        }
        auto rotated = rotate_contig_gpu(x_contig, pos_use, sin_use, cos_use,
                                         p->algo, dtype_tag, stream);
        unpack_head_vectors(rotated, out, stream);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("QY rope_unpack launch: ") +
                                     cudaGetErrorString(err));
        }
    }

    (void)kQyRopeChunkTokens;
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    RoPE::plan_dispatcher().registerDevice(Device::Type::kQy, &plan, true);
    RoPE::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    RoPE::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::rope_impl::aten
#endif
