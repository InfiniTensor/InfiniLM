#include "aclrtlaunch_add_rms_norm.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_kernel_helper.h"

namespace ascend::detail {

std::vector<at::Tensor> AddRmsNorm(const at::Tensor& x1, const at::Tensor& x2,
                                   const at::Tensor& weight, double eps) {
  // Input validation.
  TORCH_CHECK(x1.dim() > 0,
              "`AddRmsNorm`: `x1` must have at least 1 dimension.");
  TORCH_CHECK(x1.sizes() == x2.sizes(),
              "`AddRmsNorm`: `x1` and `x2` must have the same shape.");
  TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
              "`AddRmsNorm`: `x1` and `x2` must have the same dtype.");
  TORCH_CHECK(x1.scalar_type() == at::kHalf || x1.scalar_type() == at::kFloat,
              "`AddRmsNorm`: only `float16` and `float32` are supported; got ",
              x1.scalar_type(), ".");
  TORCH_CHECK(weight.dim() == 1,
              "`AddRmsNorm`: `weight` must be 1-dimensional.");
  TORCH_CHECK(weight.size(0) == x1.size(-1), "`AddRmsNorm`: `weight` size (",
              weight.size(0), ") must match input last dim (", x1.size(-1),
              ").");

  int64_t dim_length = x1.size(-1);
  int64_t total_rows = x1.numel() / dim_length;

  if (total_rows == 0 || dim_length == 0) {
    return {at::empty_like(x1), at::empty_like(x1)};
  }

  at::Tensor inp1 = x1.contiguous();
  at::Tensor inp2 = x2.contiguous();
  int64_t dtype_size = inp1.element_size();

  // Hardware parameters.
  auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  int64_t core_num = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
  uint64_t ub_size;
  ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  int64_t ub_size_limit = static_cast<int64_t>(ub_size);

  // Alignment (32-byte boundary).
  int64_t align_elements = 32 / dtype_size;
  int64_t dim_length_align =
      ((dim_length + align_elements - 1) / align_elements) * align_elements;

  // UB capacity check.
  //
  // - `fp16`: `inQ_x1` (*2*2) + `inQ_x2` (*2*2) + `outQ_y` (*2*2) +
  //   `outQ_xout` (*2*2) + `fp32Buf1` (*4) + `fp32Buf2` (*4) +
  //   `weight` (*4) = 16 + 12 = 28
  // - `fp32`: `inQ_x1` (*2*4) + `inQ_x2` (*2*4) + `outQ_y` (*2*4) +
  //   `outQ_xout` (*2*4) + `weight` (*4) = 32 + 4 = 36
  int64_t buffer_coefficient = (dtype_size == 2) ? 28 : 36;
  int64_t max_dim_length = (ub_size_limit - 1024) / buffer_coefficient;
  int64_t fp_align_elements = 32 / 4;
  max_dim_length = (max_dim_length / fp_align_elements) * fp_align_elements;
  TORCH_CHECK(dim_length_align <= max_dim_length,
              "`AddRmsNorm`: `dim_length` ", dim_length, " (aligned ",
              dim_length_align, ") exceeds UB capacity (max ", max_dim_length,
              ").");

  // Padding.
  at::Tensor kernel_input1;
  at::Tensor kernel_input2;

  if (dim_length != dim_length_align) {
    kernel_input1 = inp1.reshape({total_rows, dim_length});
    kernel_input1 = at::constant_pad_nd(
        kernel_input1, {0, dim_length_align - dim_length}, 0.0);
    kernel_input1 = kernel_input1.contiguous();

    kernel_input2 = inp2.reshape({total_rows, dim_length});
    kernel_input2 = at::constant_pad_nd(
        kernel_input2, {0, dim_length_align - dim_length}, 0.0);
    kernel_input2 = kernel_input2.contiguous();
  } else {
    kernel_input1 = inp1.reshape({total_rows, dim_length_align}).contiguous();
    kernel_input2 = inp2.reshape({total_rows, dim_length_align}).contiguous();
  }

  at::Tensor kernel_output_y = at::empty_like(kernel_input1);
  at::Tensor kernel_output_x_out = at::empty_like(kernel_input1);

  // Weight: always pass as fp32, padded to `dim_length_align`.
  at::Tensor weight_float = weight.contiguous().to(at::kFloat);

  if (dim_length != dim_length_align) {
    weight_float = at::constant_pad_nd(
        weight_float, {0, dim_length_align - dim_length}, 0.0);
  }

  weight_float = weight_float.contiguous();

  // Block-level tiling (distribute rows across cores).
  int64_t used_core_num = std::min(total_rows, core_num);
  int64_t former_length = (total_rows + used_core_num - 1) / used_core_num;
  int64_t tail_length = former_length - 1;
  int64_t former_num = total_rows - tail_length * used_core_num;
  uint32_t block_dim = static_cast<uint32_t>(used_core_num);

  // All `EXEC_KERNEL_CMD` args must be lvalues.
  float eps_float = static_cast<float>(eps);
  int64_t dtype_size_val = dtype_size;

  // The first arg `add_rms_norm` is the AscendC kernel entry-point name — it
  // must match `ascendc_add_operator(OP_NAME add_rms_norm)` in `CMakeLists.txt`,
  // the `__global__ __aicore__ void add_rms_norm(...)` definition in
  // `op_kernel/`, and the generated `aclrtlaunch_add_rms_norm.h` header.
  // Google C++ Style's PascalCase rule does NOT apply: this identifier is
  // dictated by the AscendC toolchain's symbol convention.
  EXEC_KERNEL_CMD(add_rms_norm, block_dim, kernel_input1, kernel_input2,
                  weight_float, kernel_output_y, kernel_output_x_out,
                  total_rows, dim_length, dim_length_align, former_num,
                  former_length, tail_length, eps_float, dtype_size_val);

  // Remove padding and reshape back to original shape.
  at::Tensor output_y = kernel_output_y;
  at::Tensor output_x_out = kernel_output_x_out;

  if (dim_length != dim_length_align) {
    output_y = output_y.narrow(-1, 0, dim_length).contiguous();
    output_x_out = output_x_out.narrow(-1, 0, dim_length).contiguous();
  }

  output_y = output_y.reshape(x1.sizes());
  output_x_out = output_x_out.reshape(x1.sizes());

  return {output_y, output_x_out};
}

}  // namespace ascend::detail
