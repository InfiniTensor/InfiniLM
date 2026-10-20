#ifndef INFINI_OPS_ASCEND_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ASCEND_RMS_NORM_KERNEL_H_

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_rms_norm.h"
#include "aclnnop/aclnn_cast.h"
#include "base/rms_norm.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kAscend> : public RmsNorm {
 public:
  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : RmsNorm(input, weight, eps, out), weight_cache_(weight) {
    assert(ndim_ >= 2 && "`RmsNorm` Ascend path requires rank >= 2");
    assert(input.strides().back() == 1 && out.strides().back() == 1 &&
           "`RmsNorm` Ascend path requires a contiguous last dimension");

    needs_input_staging_ = !input.IsContiguous();
    needs_out_staging_ = !out.IsContiguous();
    logical_rows_ = input.numel() / dim_;
    element_size_ = kDataTypeToSize.at(input.dtype());
    row_bytes_ = dim_ * element_size_;
    tensor_bytes_ = logical_rows_ * row_bytes_;

    std::vector<int64_t> input_shape(input.shape().begin(),
                                     input.shape().end());
    std::vector<int64_t> out_shape(out.shape().begin(), out.shape().end());
    if (needs_input_staging_) {
      norm_in_cache_ = ascend::AclTensorCache(
          input_shape, ascend::ToAclDtype(input.dtype()), nullptr);
    } else {
      norm_in_cache_ = ascend::AclTensorCache(input);
    }
    if (needs_out_staging_) {
      norm_out_cache_ = ascend::AclTensorCache(
          out_shape, ascend::ToAclDtype(out.dtype()), nullptr);
    } else {
      norm_out_cache_ = ascend::AclTensorCache(out);
    }

    needs_weight_cast_ =
        input.dtype() != weight.dtype() && weight.dtype() != DataType::kFloat32;
    if (needs_weight_cast_) {
      auto fp32_bytes = static_cast<size_t>(dim_) * sizeof(float);
      auto ret = aclrtMalloc(&weight_fp32_data_, fp32_bytes,
                             ACL_MEM_MALLOC_NORMAL_ONLY);
      assert(ret == ACL_SUCCESS &&
             "`RmsNorm` Ascend path failed to allocate cast weight");
      weight_fp32_cache_ = ascend::AclTensorCache({static_cast<int64_t>(dim_)},
                                                  ACL_FLOAT, weight_fp32_data_);
    }

    rstd_shape_.assign(input.shape().begin(), input.shape().end() - 1);
    rstd_strides_.resize(rstd_shape_.size());
    int64_t rstd_stride = 1;
    for (int64_t axis = static_cast<int64_t>(rstd_shape_.size()) - 1; axis >= 0;
         --axis) {
      rstd_strides_[axis] = rstd_stride;
      rstd_stride *= rstd_shape_[axis];
    }
    rstd_size_ = logical_rows_ * sizeof(float);

    // Follow InfiniCore's proven ACLNN path: prepare one repeatable executor
    // from stable tensor metadata. Runtime calls only rebind data addresses.
    auto t_in = norm_in_cache_.get(nullptr);
    auto t_out = norm_out_cache_.get(nullptr);
    aclTensor* t_weight;
    if (needs_weight_cast_) {
      auto t_weight_src = weight_cache_.get(nullptr);
      auto t_weight_dst = weight_fp32_cache_.get(weight_fp32_data_);
      aclnnCastGetWorkspaceSize(t_weight_src, ACL_FLOAT, t_weight_dst,
                                &cast_ws_, &cast_exec_);
      aclSetAclOpExecutorRepeatable(cast_exec_);
      t_weight = t_weight_dst;
    } else {
      t_weight = weight_cache_.get(nullptr);
    }

    rstd_tensor_ = aclCreateTensor(
        rstd_shape_.data(), static_cast<int64_t>(rstd_shape_.size()), ACL_FLOAT,
        rstd_strides_.data(), 0, ACL_FORMAT_ND, rstd_shape_.data(),
        static_cast<int64_t>(rstd_shape_.size()), nullptr);
    aclnnRmsNormGetWorkspaceSize(t_in, t_weight, eps, t_out, rstd_tensor_,
                                 &ws_size_, &executor_);
    aclSetAclOpExecutorRepeatable(executor_);
    total_workspace_size_ = ws_size_ + rstd_size_;
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    norm_in_cache_.release();
    weight_cache_.release();
    norm_out_cache_.release();
    weight_fp32_cache_.release();
    if (weight_fp32_data_) aclrtFree(weight_fp32_data_);
    // `rstd_tensor_` leaks with the executor at shutdown (see `64c367c`).
  }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    void* input_data = const_cast<void*>(input.data());
    void* out_data = out.data();

    if (needs_input_staging_) {
      auto& staging = ascend::GetWorkspacePool().Ensure(stream, tensor_bytes_,
                                                        "rms_norm_input");
      PackRows(staging.buf, input.data(), input_shape_, input_strides_, stream);
      input_data = staging.buf;
    }
    if (needs_out_staging_) {
      auto& staging = ascend::GetWorkspacePool().Ensure(stream, tensor_bytes_,
                                                        "rms_norm_output");
      out_data = staging.buf;
    }

    auto t_in = norm_in_cache_.get(input_data);
    auto t_out = norm_out_cache_.get(out_data);
    aclTensor* t_weight;
    void* weight_data;

    if (needs_weight_cast_) {
      auto t_weight_src = weight_cache_.get(const_cast<void*>(weight.data()));
      auto t_weight_dst = weight_fp32_cache_.get(weight_fp32_data_);
      AclSetTensorAddr(cast_exec_, 0, t_weight_src,
                       const_cast<void*>(weight.data()));
      AclSetTensorAddr(cast_exec_, 1, t_weight_dst, weight_fp32_data_);
      auto& cast_arena = ascend::GetWorkspacePool().Ensure(stream, cast_ws_);
      aclnnCast(cast_arena.buf, cast_ws_, cast_exec_, stream);
      t_weight = t_weight_dst;
      weight_data = weight_fp32_data_;
    } else {
      t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
      weight_data = const_cast<void*>(weight.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, total_workspace_size_, "rms_norm");
    auto* rstd_data = static_cast<uint8_t*>(arena.buf) + ws_size_;
    aclSetRawTensorAddr(rstd_tensor_, rstd_data);

    AclSetTensorAddr(executor_, 0, t_in, input_data);
    AclSetTensorAddr(executor_, 1, t_weight, weight_data);
    AclSetTensorAddr(executor_, 2, t_out, out_data);
    AclSetTensorAddr(executor_, 3, rstd_tensor_, rstd_data);
    aclnnRmsNorm(arena.buf, ws_size_, executor_, stream);

    if (needs_out_staging_) {
      UnpackRows(out.data(), out_data, out_shape_, out_strides_, stream);
    }
  }

 private:
  int64_t RowOffset(int64_t row, const Tensor::Shape& shape,
                    const Tensor::Strides& strides) const {
    int64_t remaining = row;
    int64_t offset = 0;
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 2; axis >= 0;
         --axis) {
      auto coordinate = remaining % static_cast<int64_t>(shape[axis]);
      remaining /= static_cast<int64_t>(shape[axis]);
      offset += coordinate * static_cast<int64_t>(strides[axis]);
    }
    return offset;
  }

  void PackRows(void* dst, const void* src, const Tensor::Shape& shape,
                const Tensor::Strides& strides, aclrtStream stream) const {
    for (int64_t row = 0; row < logical_rows_; ++row) {
      auto offset = RowOffset(row, shape, strides);
      auto ret = aclrtMemcpyAsync(
          static_cast<uint8_t*>(dst) + row * row_bytes_, row_bytes_,
          static_cast<const uint8_t*>(src) + offset * element_size_, row_bytes_,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      assert(ret == ACL_SUCCESS && "`RmsNorm` input pack failed");
    }
  }

  void UnpackRows(void* dst, const void* src, const Tensor::Shape& shape,
                  const Tensor::Strides& strides, aclrtStream stream) const {
    for (int64_t row = 0; row < logical_rows_; ++row) {
      auto offset = RowOffset(row, shape, strides);
      auto ret = aclrtMemcpyAsync(
          static_cast<uint8_t*>(dst) + offset * element_size_, row_bytes_,
          static_cast<const uint8_t*>(src) + row * row_bytes_, row_bytes_,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      assert(ret == ACL_SUCCESS && "`RmsNorm` output unpack failed");
    }
  }

  mutable ascend::AclTensorCache norm_in_cache_;
  mutable ascend::AclTensorCache weight_cache_;
  mutable ascend::AclTensorCache norm_out_cache_;
  mutable ascend::AclTensorCache weight_fp32_cache_;
  bool needs_input_staging_{false};
  bool needs_out_staging_{false};
  bool needs_weight_cast_{false};
  int64_t logical_rows_{0};
  uint64_t element_size_{0};
  uint64_t row_bytes_{0};
  uint64_t tensor_bytes_{0};
  void* weight_fp32_data_{nullptr};
  mutable aclOpExecutor* cast_exec_{nullptr};
  mutable uint64_t cast_ws_{0};
  mutable aclOpExecutor* executor_{nullptr};
  mutable uint64_t ws_size_{0};
  std::vector<int64_t> rstd_shape_;
  std::vector<int64_t> rstd_strides_;
  uint64_t rstd_size_{0};
  uint64_t total_workspace_size_{0};
  mutable aclTensor* rstd_tensor_{nullptr};
};

}  // namespace infini::ops

#endif
