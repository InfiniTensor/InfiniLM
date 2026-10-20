#include "aclrtlaunch_rms_norm.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_kernel_helper.h"

namespace ascend::detail {

at::Tensor RmsNorm(const at::Tensor& input, const at::Tensor& weight,
                   double eps) {
  // Input validation.
  TORCH_CHECK(input.dim() > 0,
              "`RmsNorm`: `input` must have at least 1 dimension.");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat,
      "`RmsNorm`: only `float16` and `float32` are supported; got ",
      input.scalar_type(), ".");
  TORCH_CHECK(weight.dim() == 1, "`RmsNorm`: `weight` must be 1-dimensional.");
  TORCH_CHECK(weight.size(0) == input.size(-1), "`RmsNorm`: `weight` size (",
              weight.size(0), ") must match input last dim (", input.size(-1),
              ").");

  int64_t dim_length = input.size(-1);
  int64_t total_rows = input.numel() / dim_length;

  if (total_rows == 0 || dim_length == 0) {
    return at::empty_like(input);
  }

  at::Tensor x = input.contiguous();
  int64_t dtype_size = x.element_size();

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
  // - `fp32`: `inQ` (*2) + `outQ` (*2) + `weight` = 5 * `dim_length_align`
  //   * 4 = coeff 20.
  // - `fp16`: `inQ` (*2) + `outQ` (*2) + `x_fp32` + `tmp_fp32` + `weight`
  //   = 2 * `dim_length_align` * 2 * 2 + 3 * `dim_length_align` * 4 =
  //   8 + 12 = coeff 20.
  int64_t buffer_coefficient = 20;
  // Reserve 1024 bytes for reduce buffers.
  int64_t max_dim_length = (ub_size_limit - 1024) / buffer_coefficient;
  // `fp32` alignment.
  int64_t fp_align_elements = 32 / 4;
  max_dim_length = (max_dim_length / fp_align_elements) * fp_align_elements;
  TORCH_CHECK(dim_length_align <= max_dim_length,
              "`RmsNorm`: `dim_length` ", dim_length, " (aligned ",
              dim_length_align, ") exceeds UB capacity (max ", max_dim_length,
              ").");

  // Padding.
  at::Tensor kernel_input;

  if (dim_length != dim_length_align) {
    kernel_input = x.reshape({total_rows, dim_length});
    kernel_input = at::constant_pad_nd(
        kernel_input, {0, dim_length_align - dim_length}, 0.0);
    kernel_input = kernel_input.contiguous();
  } else {
    kernel_input = x.reshape({total_rows, dim_length_align}).contiguous();
  }

  at::Tensor kernel_output = at::empty_like(kernel_input);

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

  // The first arg `rms_norm` is the AscendC kernel entry-point name — it
  // must match `ascendc_add_operator(OP_NAME rms_norm)` in `CMakeLists.txt`,
  // the `__global__ __aicore__ void rms_norm(...)` definition in `op_kernel/`,
  // and the generated `aclrtlaunch_rms_norm.h` header.  Google C++ Style's
  // PascalCase rule does NOT apply: this identifier is dictated by the
  // AscendC toolchain's symbol convention.
  EXEC_KERNEL_CMD(rms_norm, block_dim, kernel_input, weight_float,
                  kernel_output, total_rows, dim_length, dim_length_align,
                  former_num, former_length, tail_length, eps_float,
                  dtype_size_val);

  // Remove padding and reshape back to original shape.
  at::Tensor output = kernel_output;

  if (dim_length != dim_length_align) {
    output = output.narrow(-1, 0, dim_length).contiguous();
  }

  output = output.reshape(input.sizes());

  return output;
}

}  // namespace ascend::detail
