#include "infinicore/ops/baddbmm.hpp"
#include "infinicore/ops/gemm.hpp"
#include "infinicore/ops/rearrange.hpp"

namespace infinicore::op {

// 内联的 BLAS 兼容性检查，减少函数调用开销
inline bool is_blas_compatible(const Tensor &t) {
    const auto ndim = t->ndim();
    if (ndim == 2) {
        const auto rs = t->stride(0);
        const auto cs = t->stride(1);
        if (rs != 1 && cs != 1) {
            return false;
        }
        if (rs == 1 && cs == 1) {
            return t->shape()[0] == 1 || t->shape()[1] == 1;
        }
        return true;
    } else if (ndim == 3) {
        const auto rs = t->stride(1);
        const auto cs = t->stride(2);
        if (t->shape()[0] > 1 && t->stride(0) == 0) {
            return false;
        }
        if (rs != 1 && cs != 1) {
            return false;
        }
        if (rs == 1 && cs == 1) {
            return t->shape()[1] == 1 || t->shape()[2] == 1;
        }
        return true;
    }
    return false;
}

inline void prepare_gemm_input(Tensor &output, Tensor &input, const size_t batch_size, const size_t m, const size_t n) {
    const auto input_ndim = input->ndim();
    if (input_ndim == 2) {
        rearrange_(output, input->as_strided(
                               {batch_size, m, n},
                               {0, input->stride(0), input->stride(1)}));
    } else if (input_ndim == 3 && input->shape()[0] == 1 && batch_size > 1) {
        rearrange_(output, input->as_strided(
                               {batch_size, m, n},
                               {0, input->stride(1), input->stride(2)}));
    } else {
        rearrange_(output, input);
    }
}

Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2,
               float beta,
               float alpha) {
    const size_t batch_size = batch1->shape()[0];
    const size_t m = batch1->shape()[1];
    const size_t n = batch2->shape()[2];

    const Tensor &a = is_blas_compatible(batch1) ? batch1 : rearrange(batch1);
    const Tensor &b = is_blas_compatible(batch2) ? batch2 : rearrange(batch2);

    if (beta == 0.0f) {
        return gemm(a, b, alpha, 0.0f);
    }

    Tensor result = Tensor::empty({batch_size, m, n}, a->dtype(), a->device());

    prepare_gemm_input(result, input, batch_size, m, n);

    gemm_(result, a, b, alpha, beta);
    return result;
}

void baddbmm_(Tensor out, Tensor input, Tensor batch1, Tensor batch2,
              float beta,
              float alpha) {
    const size_t batch_size = batch1->shape()[0];
    const size_t m = batch1->shape()[1];
    const size_t n = batch2->shape()[2];

    const Tensor &a = is_blas_compatible(batch1) ? batch1 : rearrange(batch1);
    const Tensor &b = is_blas_compatible(batch2) ? batch2 : rearrange(batch2);

    const bool out_is_usable = out->is_contiguous() && out->ndim() == 3 && out->shape()[0] == batch_size && out->shape()[1] == m && out->shape()[2] == n;

    if (out_is_usable) {
        if (beta != 0.0f && input->data() != out->data()) {
            prepare_gemm_input(out, input, batch_size, m, n);
        }
        gemm_(out, a, b, alpha, beta);
    } else {
        Tensor result = Tensor::empty({batch_size, m, n}, a->dtype(), a->device());
        if (beta != 0.0f) {
            prepare_gemm_input(result, input, batch_size, m, n);
        }
        gemm_(result, a, b, alpha, beta);
        rearrange_(out, result);
    }
}
} // namespace infinicore::op
