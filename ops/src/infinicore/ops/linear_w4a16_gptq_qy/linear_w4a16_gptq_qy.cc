#include "infinicore/ops/linear_w4a16_gptq_qy.hpp"
#include "infinicore/ops/scaled_mm_w4a16_gptq_qy.hpp"
#include <iostream>
namespace infinicore::op {

Tensor linear_w4a16_gptq_qy(Tensor input, Tensor qweight, Tensor qzeros, Tensor scales, int64_t quant_type, int64_t bit) {

    Size ndim = input->ndim();

    Size out_features = qweight->shape()[1];

    // 2. 计算输出形状 [..., out_features]
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;

    // 3. 分配输出显存
    auto out = Tensor::zeros(output_shape, input->dtype(), input->device());

    // 4. 执行计算
    linear_w4a16_gptq_qy_(out, input, qweight, scales, qzeros, quant_type, bit);

    return out;
}

void linear_w4a16_gptq_qy_(Tensor out, Tensor in, Tensor qweights, Tensor scales, Tensor qzeros, int64_t quant_type, int64_t bit) {

    Size in_features = qweights->shape()[0] * 2; // ✅ 修正：第 0 维是 in/2
    Size out_features = qweights->shape()[1];    // ✅ 修正：第 1 维是 out

    // 检查输入输出维度
    Size ndim = in->ndim();

    // ========================================================================
    // 合并 Batch 维度
    // ========================================================================
    Size N = 1;
    auto input_shape = in->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    op::scaled_mm_w4a16_gptq_qy_(
        out->view({N, out_features}), // Output: [N, out]
        in->view({N, in_features}),   // Input:  [N, in]
        qweights,                     // Weight: [in/2, out]
        scales,                       // Scales: [in/group, out]
        qzeros,                       // QZeros: [in/group, out]
        quant_type,                   // Quantization type
        bit                           // Bit width
    );
    // out->debug();
}

} // namespace infinicore::op
