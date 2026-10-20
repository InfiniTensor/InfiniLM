#ifndef INFINI_OPS_CAMBRICON_ADD_H_
#define INFINI_OPS_CAMBRICON_ADD_H_

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "base/add.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T>
void AddUnion(void* workspace, int core_per_cluster, int cluster_count,
              cnrtQueue_t queue, const void* input, const void* other,
              void* out, const size_t* out_shape, const ptrdiff_t* out_strides,
              const size_t* input_shape, const ptrdiff_t* input_strides,
              const size_t* other_shape, const ptrdiff_t* other_strides,
              size_t output_size, int ndim, bool fast_path, bool out_contiguous,
              double alpha);

template <>
class Operator<Add, Device::Type::kCambricon> : public Add {
 public:
  Operator(const Tensor input, const Tensor other, const double alpha,
           Tensor out)
      : Add{input, other, alpha, out} {
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster,
                                &cluster_count);
    cnrtMalloc(&default_workspace_, workspace_size_in_bytes());
    PrepareMaskAdd(input, other, alpha, out);
  }

  Operator(const Tensor input, const Tensor other, Tensor out)
      : Operator{input, other, 1.0, out} {}

  void operator()(const Tensor input, const Tensor other, const double alpha,
                  Tensor out) const override {
    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    auto workspace{workspace_ ? workspace_ : default_workspace_};

    if (mask_op_ && out.shape() == out_shape_ && CanUseMaskAdd(input, other, alpha, out)) {
      RunMaskAdd(queue, input, other, out);
      return;
    }

    bool fast_path = is_input_contiguous_ && is_other_contiguous_ &&
                     is_out_contiguous_ && input_shape_ == out_shape_ &&
                     other_shape_ == out_shape_;

    DispatchFunc<Device::Type::kCambricon,
                 List<DataType::kFloat16, DataType::kBFloat16,
                      DataType::kFloat32, DataType::kInt32, DataType::kInt64>>(
        {out_type_},
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          AddUnion<T>(workspace, core_per_cluster, cluster_count, queue,
                      input.data(), other.data(), out.data(), out_shape_.data(),
                      out_strides_.data(), input_shape_.data(),
                      input_strides_.data(), other_shape_.data(),
                      other_strides_.data(), output_size_, ndim_, fast_path,
                      is_out_contiguous_, alpha);
        },
        "CambriconAdd::operator() - output dispatch");
  }

  ~Operator() { cnrtFree(default_workspace_); }

  std::size_t workspace_size_in_bytes() const override {
    return ndim_ * (3 * sizeof(size_t) + 3 * sizeof(ptrdiff_t));
  }

  void* default_workspace_{nullptr};
  int core_per_cluster = 0;
  int cluster_count = 0;

 private:
  // This leaf uses the same prepared-owner lifetime as the upstream Softmax:
  // caller completion must precede owner/cache release and stream destruction.
  // Capture remains opt-in false through OperatorBase.
  struct MaskOpDeleter {
    void operator()(cnnlOpTensorDescriptor_t p) const noexcept {
      if (p) (void)cnnlDestroyOpTensorDescriptor(p);
    }
  };
  using MaskOp = std::unique_ptr<std::remove_pointer_t<cnnlOpTensorDescriptor_t>,
                                MaskOpDeleter>;
  static void CheckMaskStatus(cnnlStatus_t status, const char* operation) {
    if (status != CNNL_STATUS_SUCCESS) {
      std::fprintf(stderr, "MASK_ADD_CNNL_FAIL %s %d\n", operation, int(status));
      std::abort();
    }
  }
  static bool CanUseMaskAdd(const Tensor input, const Tensor other,
                            double alpha, const Tensor out) {
    if (input.dtype() != DataType::kFloat32 || other.dtype() != DataType::kFloat32 ||
        out.dtype() != DataType::kFloat32 || alpha != 1.0 ||
        input.device() != out.device() || other.device() != out.device() ||
        out.ndim() != 3 || input.shape() != out.shape() || other.ndim() != 2 ||
        !input.IsContiguous() || !other.IsContiguous() || !out.IsContiguous() ||
        !input.data() || !other.data() || input.data() != out.data()) return false;
    const auto h = out.size(0), r = out.size(1), c = out.size(2);
    if (h < 2 || h > INT_MAX || r < 2 || r > INT_MAX || c < 2 || c > INT_MAX ||
        other.size(0) != r || other.size(1) != c || r > size_t(INT_MAX) / c)
      return false;
    const auto plane = r * c;
    if (h > size_t(INT_MAX) / plane) return false;
    const auto a = reinterpret_cast<uintptr_t>(input.data());
    const auto b = reinterpret_cast<uintptr_t>(other.data());
    if (a % sizeof(float) || b % sizeof(float)) return false;
    return b > a ? b - a >= h * plane * sizeof(float)
                 : a - b >= plane * sizeof(float);
  }
  void PrepareMaskAdd(const Tensor input, const Tensor other, double alpha,
                      const Tensor out) {
    if (!CanUseMaskAdd(input, other, alpha, out)) return;
    cnnlHandle_t raw_handle = nullptr;
    auto status = cnnlCreate(&raw_handle); mask_handle_.reset(raw_handle);
    CheckMaskStatus(status, "create handle");
    auto create_desc = [](cnnl_utils::TensorDescriptor& owner) {
      cnnlTensorDescriptor_t raw = nullptr;
      const auto result = cnnlCreateTensorDescriptor(&raw); owner.reset(raw);
      CheckMaskStatus(result, "create tensor descriptor");
    };
    create_desc(mask_input_desc_); create_desc(mask_plane_desc_); create_desc(mask_out_desc_);
    const int dims[3] = {int(out.size(0)), int(out.size(1)), int(out.size(2))};
    const int plane[3] = {1, dims[1], dims[2]};
    CheckMaskStatus(cnnlSetTensorDescriptor(mask_input_desc_.get(), CNNL_LAYOUT_ARRAY,
                    CNNL_DTYPE_FLOAT, 3, dims), "input descriptor");
    CheckMaskStatus(cnnlSetTensorDescriptor(mask_plane_desc_.get(), CNNL_LAYOUT_ARRAY,
                    CNNL_DTYPE_FLOAT, 3, plane), "plane descriptor");
    CheckMaskStatus(cnnlSetTensorDescriptor(mask_out_desc_.get(), CNNL_LAYOUT_ARRAY,
                    CNNL_DTYPE_FLOAT, 3, dims), "output descriptor");
    cnnlOpTensorDescriptor_t raw_op = nullptr;
    status = cnnlCreateOpTensorDescriptor(&raw_op); mask_op_.reset(raw_op);
    CheckMaskStatus(status, "create operation descriptor");
    CheckMaskStatus(cnnlSetOpTensorDescriptor(mask_op_.get(), CNNL_OP_TENSOR_ADD,
                    CNNL_DTYPE_FLOAT, CNNL_PROPAGATE_NAN), "operation descriptor");
    CheckMaskStatus(cnnlGetOpTensorWorkspaceSize(mask_handle_.get(), mask_input_desc_.get(),
                    mask_plane_desc_.get(), mask_out_desc_.get(), &mask_workspace_bytes_),
                    "workspace query");
    if (mask_workspace_bytes_) {
      void* raw = nullptr; const auto result = cnrtMalloc(&raw, mask_workspace_bytes_);
      mask_workspace_.reset(raw);
      if (result != cnrtSuccess) {
        std::fprintf(stderr, "MASK_ADD_CNRT_FAIL allocate %d\n", int(result)); std::abort();
      }
    }
  }
  void RunMaskAdd(cnrtQueue_t queue, const Tensor input, const Tensor other,
                  Tensor out) const {
    CheckMaskStatus(cnnlSetQueue(mask_handle_.get(), queue), "set queue");
    static const float one = 1.0F, zero = 0.0F;
    const auto status = cnnlOpTensor(mask_handle_.get(), mask_op_.get(),
        &one, mask_input_desc_.get(), input.data(), &one, mask_plane_desc_.get(),
        other.data(), mask_workspace_.get(), mask_workspace_bytes_,
        &zero, mask_out_desc_.get(), out.data());
    if (status != CNNL_STATUS_SUCCESS) {
      // A failed call can have queued partial work. Never unwind its resources.
      const auto completion = cnrtQueueSync(queue);
      std::fprintf(stderr, "MASK_ADD_ENQUEUE_FAIL cnnl=%d drain=%d\n",
                   int(status), int(completion));
      std::abort();
    }
  }
  cnnl_utils::Handle mask_handle_{};
  cnnl_utils::TensorDescriptor mask_input_desc_{}, mask_plane_desc_{}, mask_out_desc_{};
  MaskOp mask_op_{};
  cnrt_utils::DeviceBuffer mask_workspace_{};
  size_t mask_workspace_bytes_{0};
};

}  // namespace infini::ops

#endif
