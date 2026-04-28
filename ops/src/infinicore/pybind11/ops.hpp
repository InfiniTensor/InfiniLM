#pragma once

#include <pybind11/pybind11.h>

#include "ops/acos.hpp"
#include "ops/adaptive_avg_pool1d.hpp"
#include "ops/adaptive_avg_pool3d.hpp"
#include "ops/adaptive_max_pool1d.hpp"
#include "ops/add.hpp"
#include "ops/add_rms_norm.hpp"
#include "ops/addbmm.hpp"
#include "ops/addcmul.hpp"
#include "ops/addr.hpp"
#include "ops/affine_grid.hpp"
#include "ops/all.hpp"
#include "ops/argwhere.hpp"
#include "ops/asin.hpp"
#include "ops/asinh.hpp"
#include "ops/atanh.hpp"
#include "ops/attention.hpp"
#include "ops/avg_pool1d.hpp"
#include "ops/baddbmm.hpp"
#include "ops/bilinear.hpp"
#include "ops/binary_cross_entropy_with_logits.hpp"
#include "ops/bitwise_right_shift.hpp"
#include "ops/block_diag.hpp"
#include "ops/broadcast_to.hpp"
#include "ops/cat.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/cdist.hpp"
#include "ops/cross_entropy.hpp"
#include "ops/diff.hpp"
#include "ops/digamma.hpp"
#include "ops/dist.hpp"
#include "ops/embedding.hpp"
#include "ops/equal.hpp"
#include "ops/flash_attention.hpp"
#include "ops/flipud.hpp"
#include "ops/float_power.hpp"
#include "ops/floor.hpp"
#include "ops/floor_divide.hpp"
#include "ops/fmin.hpp"
#include "ops/fmod.hpp"
#include "ops/gaussian_nll_loss.hpp"
#include "ops/hardswish.hpp"
#include "ops/hardtanh.hpp"
#include "ops/hinge_embedding_loss.hpp"
#include "ops/huber_loss.hpp"
#include "ops/hypot.hpp"
#include "ops/index_add.hpp"
#include "ops/index_copy.hpp"
#include "ops/inner.hpp"
#include "ops/interpolate.hpp"
#include "ops/kron.hpp"
#include "ops/kthvalue.hpp"
#include "ops/kv_caching.hpp"
#include "ops/layer_norm.hpp"
#include "ops/ldexp.hpp"
#include "ops/lerp.hpp"
#include "ops/linear.hpp"
#include "ops/linear_w8a8i8.hpp"
#include "ops/log_softmax.hpp"
#include "ops/logaddexp.hpp"
#include "ops/logaddexp2.hpp"
#include "ops/logcumsumexp.hpp"
#include "ops/logdet.hpp"
#include "ops/logical_and.hpp"
#include "ops/logical_not.hpp"
#include "ops/masked_select.hpp"
#include "ops/matmul.hpp"
#include "ops/mha_kvcache.hpp"
#include "ops/mha_varlen.hpp"
#include "ops/mul.hpp"
#include "ops/multi_margin_loss.hpp"
#include "ops/pad.hpp"
#include "ops/paged_attention.hpp"
#include "ops/paged_attention_prefill.hpp"
#include "ops/paged_caching.hpp"
#include "ops/prelu.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/reciprocal.hpp"
#include "ops/relu6.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/scatter.hpp"
#include "ops/selu.hpp"
#include "ops/silu.hpp"
#include "ops/silu_and_mul.hpp"
#include "ops/sinh.hpp"
#include "ops/smooth_l1_loss.hpp"
#include "ops/softplus.hpp"
#include "ops/softsign.hpp"
#include "ops/sum.hpp"
#include "ops/swiglu.hpp"
#include "ops/take.hpp"
#include "ops/tan.hpp"
#include "ops/tanhshrink.hpp"
#include "ops/topk.hpp"
#include "ops/triplet_margin_loss.hpp"
#include "ops/triplet_margin_with_distance_loss.hpp"
#include "ops/unfold.hpp"
#include "ops/upsample_bilinear.hpp"
#include "ops/upsample_nearest.hpp"
#include "ops/vander.hpp"
#include "ops/var.hpp"
#include "ops/var_mean.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_adaptive_max_pool1d(m);
    bind_add(m);
    bind_adaptive_avg_pool3d(m);
    bind_argwhere(m);
    bind_addr(m);
    bind_asin(m);
    bind_add_rms_norm(m);
    bind_add(m);
    bind_addbmm(m);
    bind_acos(m);
    bind_affine_grid(m);
    bind_floor(m);
    bind_adaptive_avg_pool1d(m);
    bind_attention(m);
    bind_asinh(m);
    bind_baddbmm(m);
    bind_bilinear(m);
    bind_block_diag(m);
    bind_bitwise_right_shift(m);
    bind_causal_softmax(m);
    bind_diff(m);
    bind_digamma(m);
    bind_dist(m);
    bind_flash_attention(m);
    bind_hinge_embedding_loss(m);
    bind_kv_caching(m);
    bind_fmod(m);
    bind_fmin(m);
    bind_cat(m);
    bind_causal_softmax(m);
    bind_inner(m);
    bind_random_sample(m);
    bind_masked_select(m);
    bind_log_softmax(m);
    bind_logaddexp(m);
    bind_logaddexp2(m);
    bind_linear(m);
    bind_logdet(m);
    bind_matmul(m);
    bind_kron(m);
    bind_mul(m);
    bind_mha_kvcache(m);
    bind_mha_varlen(m);
    bind_hardswish(m);
    bind_hardtanh(m);
    bind_gaussian_nll_loss(m);
    bind_interpolate(m);
    bind_paged_attention(m);
    bind_paged_attention_prefill(m);
    bind_paged_caching(m);
    bind_pad(m);
    bind_prelu(m);
    bind_random_sample(m);
    bind_cross_entropy(m);
    bind_hypot(m);
    bind_take(m);
    bind_index_copy(m);
    bind_index_add(m);
    bind_smooth_l1_loss(m);
    bind_rearrange(m);
    bind_relu6(m);
    bind_rms_norm(m);
    bind_avg_pool1d(m);
    bind_silu(m);
    bind_swiglu(m);
    bind_tan(m);
    bind_tanhshrink(m);
    bind_logcumsumexp(m);
    bind_logical_and(m);
    bind_logical_not(m);
    bind_vander(m);
    bind_unfold(m);
    bind_rope(m);
    bind_floor_divide(m);
    bind_float_power(m);
    bind_flipud(m);
    bind_multi_margin_loss(m);
    bind_scatter(m);
    bind_broadcast_to(m);
    bind_softplus(m);
    bind_softsign(m);
    bind_linear(m);
    bind_huber_loss(m);
    bind_triplet_margin_with_distance_loss(m);
    bind_upsample_nearest(m);
    bind_embedding(m);
    bind_linear_w8a8i8(m);
    bind_silu_and_mul(m);
    bind_sum(m);
    bind_var_mean(m);
    bind_var(m);
    bind_topk(m);
    bind_all(m);
    bind_equal(m);
    bind_atanh(m);
    bind_addcmul(m);
    bind_cdist(m);
    bind_binary_cross_entropy_with_logits(m);
    bind_reciprocal(m);
    bind_upsample_bilinear(m);
    bind_kthvalue(m);
    bind_ldexp(m);
    bind_lerp(m);
    bind_triplet_margin_loss(m);
    bind_selu(m);
    bind_sinh(m);
    bind_layer_norm(m);
}

} // namespace infinicore::ops
