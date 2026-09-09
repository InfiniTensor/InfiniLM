#include "infinicore/ops/bilinear.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/rearrange.hpp"

namespace infinicore::op {

namespace {
inline bool is_gemm_compatible_3d(const Tensor &t) {
    if (t->ndim() != 3) {
        return false;
    }

    const auto batch = t->shape()[0];
    const auto rows = t->shape()[1];
    const auto cols = t->shape()[2];
    const auto bs = t->stride(0);
    const auto rs = t->stride(1);
    const auto cs = t->stride(2);

    if (rs != 1 && cs != 1) {
        return false;
    }

    if (cs == 1) {
        if (rs < static_cast<int64_t>(cols)) {
            return false;
        }
    } else {
        if (cs < static_cast<int64_t>(rows)) {
            return false;
        }
    }

    if (batch > 1 && bs == 0) {
        return false;
    }

    return true;
}

inline Tensor ensure_gemm_compatible(const Tensor &t) {
    if (t->ndim() == 2) {
        return t->is_contiguous() ? t : rearrange(t);
    } else if (t->ndim() == 3) {
        return is_gemm_compatible_3d(t) ? t : rearrange(t);
    }
    return t->is_contiguous() ? t : rearrange(t);
}

} // anonymous namespace

Tensor bilinear(Tensor x1, Tensor x2, Tensor weight, std::optional<Tensor> bias) {
    const size_t batch_size = x1->shape()[0];
    const size_t in1_features = x1->shape()[1];
    const size_t in2_features = x2->shape()[1];
    const size_t out_features = weight->shape()[0];

    Tensor x1_compat = ensure_gemm_compatible(x1);
    Tensor x2_compat = ensure_gemm_compatible(x2);
    Tensor weight_cont = weight->is_contiguous() ? weight : weight->contiguous();

    Tensor weight_permuted = weight_cont->permute({1, 0, 2});
    Tensor weight_permuted_cont = weight_permuted->is_contiguous()
                                    ? weight_permuted
                                    : weight_permuted->contiguous();
    Tensor weight_matrix = weight_permuted_cont->view({in1_features, out_features * in2_features});

    Tensor intermediate = matmul(x1_compat, weight_matrix, 1.0f);

    Tensor intermediate_3d = intermediate->view({batch_size, out_features, in2_features});
    Tensor intermediate_transposed = intermediate_3d->permute({0, 2, 1});
    Tensor intermediate_compat = ensure_gemm_compatible(intermediate_transposed);

    Tensor x2_row = x2_compat->view({batch_size, 1, in2_features});
    Tensor x2_row_compat = ensure_gemm_compatible(x2_row);

    Tensor out_3d = matmul(x2_row_compat, intermediate_compat, 1.0f);
    Tensor out = out_3d->view({batch_size, out_features});

    if (bias) {
        Tensor bias_broadcast = (*bias)->as_strided(
            {batch_size, out_features},
            {0, (*bias)->strides()[0]});
        out = add(out, bias_broadcast);
    }
    return out;
}

void bilinear_(Tensor out, Tensor x1, Tensor x2, Tensor weight, std::optional<Tensor> bias) {
    Tensor result = bilinear(x1, x2, weight, bias);
    rearrange_(out, result);
}

} // namespace infinicore::op
