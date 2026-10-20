#include "linked/tvm_ffi/moore/ops/flash_attn_with_kvcache/mate.h"

#include <tvm/ffi/container/array.h>

#include <cmath>
#include <stdexcept>

#include "linked/tvm_ffi/moore/mate.h"
#include "torch/moore/c10.h"
#include "torch/tensor_.h"

namespace infini::ops {
namespace {

namespace mate = linked::tvm_ffi::moore;
using mate::OptionalTensorView;
namespace py = mate::py;

py::object PythonTensor(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? py::cast(*tensor) : py::none();
}

void CallCombine(mate::MateFmhaRuntime& runtime,
                 mate::detail::DlPackTensor& out,
                 mate::detail::DlPackTensor& lse,
                 const tvm::ffi::Array<tvm::ffi::Tensor>& accumulators,
                 int max_seqlen_q, int num_splits) {
  auto* combine = runtime.Combine();
  if (num_splits > 1 && combine == nullptr) {
    throw std::runtime_error(
        "MATE selected split-KV execution without a combine kernel");
  }
  if (num_splits <= 1) return;

  tvm::ffi::Function::InvokeExternC(
      nullptr, combine->Entry(), OptionalTensorView{}, OptionalTensorView{},
      tvm::ffi::Optional<int>{max_seqlen_q}, tvm::ffi::TensorView(out.Get()),
      tvm::ffi::TensorView(lse.Get()), tvm::ffi::TensorView(accumulators[0]),
      tvm::ffi::TensorView(accumulators[1]), OptionalTensorView{},
      int{num_splits});
}

}  // namespace

struct Operator<FlashAttnWithKvcache, Device::Type::kMoore, 16>::MateState {
  linked::tvm_ffi::moore::MateFmhaRuntime runtime_;
  std::optional<at::Tensor> scalar_cache_seqlens_;
  std::optional<at::Tensor> internal_lse_;
};

Operator<FlashAttnWithKvcache, Device::Type::kMoore, 16>::~Operator() = default;

void Operator<FlashAttnWithKvcache, Device::Type::kMoore, 16>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin, const int64_t cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  Run(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, std::nullopt,
      cache_seqlens, cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
      softmax_scale, causal, window_size, softcap, rotary_interleaved,
      num_splits, return_softmax_lse, out, softmax_lse);
}

void Operator<FlashAttnWithKvcache, Device::Type::kMoore, 16>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin,
    const std::optional<Tensor> cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  Run(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
      std::nullopt, cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
      softmax_scale, causal, window_size, softcap, rotary_interleaved,
      num_splits, return_softmax_lse, out, softmax_lse);
}

void Operator<FlashAttnWithKvcache, Device::Type::kMoore, 16>::Run(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin,
    const std::optional<Tensor> cache_seqlens,
    const std::optional<int64_t> scalar_cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  (void)return_softmax_lse;
  TORCH_CHECK(!alibi_slopes.has_value(), "MATE ALiBi is not supported");

  std::lock_guard lock{runtime_mutex_};
  if (!state_) state_ = std::make_shared<MateState>();
  auto& state = *state_;
  const typename C10<Device::Type::kMoore>::StreamGuard stream_guard{
      C10<Device::Type::kMoore>::GetStreamFromExternal(stream_, device_index_)};

  auto at_q =
      ToAtenTensor<Device::Type::kMoore>(const_cast<void*>(q.data()), q_shape_,
                                         q_strides_, q_dtype_, device_index_);
  auto at_k_cache = ToAtenTensor<Device::Type::kMoore>(
      k_cache.data(), k_cache_shape_, k_cache_strides_, k_cache_dtype_,
      device_index_);
  auto at_v_cache = ToAtenTensor<Device::Type::kMoore>(
      v_cache.data(), v_cache_shape_, v_cache_strides_, v_cache_dtype_,
      device_index_);
  auto at_out = ToAtenTensor<Device::Type::kMoore>(
      out.data(), out_shape_, out_strides_, out_dtype_, device_index_);

  auto optional_tensor = [&](const std::optional<Tensor>& tensor,
                             const Tensor::Shape& shape,
                             const Tensor::Strides& strides, DataType dtype) {
    std::optional<at::Tensor> result;
    if (tensor.has_value()) {
      result.emplace(ToAtenTensor<Device::Type::kMoore>(
          const_cast<void*>(tensor->data()), shape, strides, dtype,
          device_index_));
    }
    return result;
  };

  const auto at_k = optional_tensor(k, k_shape_, k_strides_, k_dtype_);
  const auto at_v = optional_tensor(v, v_shape_, v_strides_, v_dtype_);
  const auto at_rotary_cos = optional_tensor(
      rotary_cos, rotary_cos_shape_, rotary_cos_strides_, rotary_cos_dtype_);
  const auto at_rotary_sin = optional_tensor(
      rotary_sin, rotary_sin_shape_, rotary_sin_strides_, rotary_sin_dtype_);
  auto at_cache_seqlens =
      optional_tensor(cache_seqlens, cache_seqlens_shape_,
                      cache_seqlens_strides_, cache_seqlens_dtype_);
  const auto at_cache_batch_idx =
      optional_tensor(cache_batch_idx, cache_batch_idx_shape_,
                      cache_batch_idx_strides_, cache_batch_idx_dtype_);
  const auto at_cache_leftpad =
      optional_tensor(cache_leftpad, cache_leftpad_shape_,
                      cache_leftpad_strides_, cache_leftpad_dtype_);
  const auto at_block_table =
      optional_tensor(block_table, block_table_shape_, block_table_strides_,
                      block_table_dtype_);

  if (!at_cache_seqlens.has_value() && scalar_cache_seqlens.has_value()) {
    state.scalar_cache_seqlens_.emplace(
        at::full({static_cast<int64_t>(batch_size_)},
                 static_cast<int64_t>(*scalar_cache_seqlens),
                 at_out.options().dtype(at::kInt)));
    at_cache_seqlens = state.scalar_cache_seqlens_;
  }
  at::Tensor at_lse;
  if (softmax_lse.has_value()) {
    at_lse = ToAtenTensor<Device::Type::kMoore>(
        const_cast<void*>(softmax_lse->data()), softmax_lse_shape_,
        softmax_lse_strides_, softmax_lse_dtype_, device_index_);
  } else {
    if (!state.internal_lse_.has_value()) {
      state.internal_lse_.emplace(at::empty(
          {static_cast<int64_t>(q_shape_[0]), static_cast<int64_t>(q_shape_[2]),
           static_cast<int64_t>(q_shape_[1])},
          at_out.options().dtype(at::kFloat)));
    }
    at_lse = *state.internal_lse_;
  }
  const auto max_seqlen_q = static_cast<int>(q_shape_[1]);
  const auto pack_gqa = q_shape_[2] != k_cache_shape_[2];

  if (state.runtime_.Forward().Entry() == nullptr) {
    py::gil_scoped_acquire gil;
    try {
      mate::detail::ModuleRecorder recorder;
      recorder.Forward().attr("_fmha_fwd")(
          py::arg("q") = at_q, py::arg("k") = at_k_cache,
          py::arg("v") = at_v_cache, py::arg("k_new") = PythonTensor(at_k),
          py::arg("v_new") = PythonTensor(at_v), py::arg("q_v") = py::none(),
          py::arg("cu_seqlens_q") = py::none(),
          py::arg("cu_seqlens_k") = py::none(),
          py::arg("cu_seqlens_k_new") = py::none(),
          py::arg("seqused_q") = py::none(),
          py::arg("seqused_k") = PythonTensor(at_cache_seqlens),
          py::arg("max_seqlen_q") = max_seqlen_q,
          py::arg("max_seqlen_k") = py::none(),
          py::arg("page_table") = PythonTensor(at_block_table),
          py::arg("kv_batch_idx") = PythonTensor(at_cache_batch_idx),
          py::arg("leftpad_k") = PythonTensor(at_cache_leftpad),
          py::arg("rotary_cos") = PythonTensor(at_rotary_cos),
          py::arg("rotary_sin") = PythonTensor(at_rotary_sin),
          py::arg("seqlens_rotary") = py::none(),
          py::arg("q_descale") = py::none(), py::arg("k_descale") = py::none(),
          py::arg("v_descale") = py::none(),
          py::arg("softmax_scale") = static_cast<float>(softmax_scale.value_or(
              1.0 / std::sqrt(static_cast<double>(head_size_)))),
          py::arg("is_causal") = causal,
          py::arg("window_size_left") = window_size[0],
          py::arg("window_size_right") = window_size[1],
          py::arg("attention_chunk") = 0,
          py::arg("learnable_sink") = py::none(),
          py::arg("softcap") = static_cast<float>(softcap),
          py::arg("is_rotary_interleaved") = rotary_interleaved,
          py::arg("scheduler_metadata") = py::none(),
          py::arg("num_splits") = num_splits, py::arg("pack_gqa") = pack_gqa,
          py::arg("mp_margin") = 0, py::arg("return_lse") = true,
          py::arg("lse") = at_lse, py::arg("out") = at_out,
          py::arg("cp_world_size") = 1, py::arg("cp_rank") = 0,
          py::arg("cp_tot_seqused_k") = py::none(), py::arg("only_qv") = false);
      state.runtime_.Load(recorder.ForwardName(), recorder.CombineName());
    } catch (const py::error_already_set& error) {
      TORCH_CHECK(false, "MATE flash_attn_with_kvcache bootstrap failed: ",
                  error.what());
    }
    return;
  }

  mate::detail::DlPackTensor dl_q{at_q};
  mate::detail::DlPackTensor dl_k_cache{at_k_cache};
  mate::detail::DlPackTensor dl_v_cache{at_v_cache};
  mate::detail::DlPackTensor dl_out{at_out};
  mate::detail::DlPackTensor dl_lse{at_lse};
  std::optional<mate::detail::DlPackTensor> dl_k;
  std::optional<mate::detail::DlPackTensor> dl_v;
  std::optional<mate::detail::DlPackTensor> dl_rotary_cos;
  std::optional<mate::detail::DlPackTensor> dl_rotary_sin;
  std::optional<mate::detail::DlPackTensor> dl_cache_seqlens;
  std::optional<mate::detail::DlPackTensor> dl_cache_batch_idx;
  std::optional<mate::detail::DlPackTensor> dl_cache_leftpad;
  std::optional<mate::detail::DlPackTensor> dl_block_table;
  if (at_k.has_value()) dl_k.emplace(*at_k);
  if (at_v.has_value()) dl_v.emplace(*at_v);
  if (at_rotary_cos.has_value()) dl_rotary_cos.emplace(*at_rotary_cos);
  if (at_rotary_sin.has_value()) dl_rotary_sin.emplace(*at_rotary_sin);
  if (at_cache_seqlens.has_value()) {
    dl_cache_seqlens.emplace(*at_cache_seqlens);
  }
  if (at_cache_batch_idx.has_value()) {
    dl_cache_batch_idx.emplace(*at_cache_batch_idx);
  }
  if (at_cache_leftpad.has_value()) {
    dl_cache_leftpad.emplace(*at_cache_leftpad);
  }
  if (at_block_table.has_value()) dl_block_table.emplace(*at_block_table);

  const mate::detail::TvmStreamGuard tvm_stream_guard{
      dl_q.Get()->device,
      C10<Device::Type::kMoore>::GetStreamFromExternal(stream_, device_index_)};

  auto optional_view =
      [](const std::optional<mate::detail::DlPackTensor>& tensor) {
        return tensor.has_value()
                   ? OptionalTensorView{tvm::ffi::TensorView(tensor->Get())}
                   : OptionalTensorView{};
      };

  auto result = tvm::ffi::Function::InvokeExternC(
      nullptr, state.runtime_.Forward().Entry(),
      tvm::ffi::TensorView(dl_q.Get()), tvm::ffi::TensorView(dl_k_cache.Get()),
      tvm::ffi::TensorView(dl_v_cache.Get()), optional_view(dl_k),
      optional_view(dl_v), OptionalTensorView{}, OptionalTensorView{},
      OptionalTensorView{}, OptionalTensorView{},
      optional_view(dl_cache_seqlens), tvm::ffi::Optional<int>{max_seqlen_q},
      tvm::ffi::Optional<int>{}, optional_view(dl_block_table),
      optional_view(dl_cache_batch_idx), optional_view(dl_cache_leftpad),
      optional_view(dl_rotary_cos), optional_view(dl_rotary_sin),
      OptionalTensorView{}, OptionalTensorView{}, OptionalTensorView{},
      OptionalTensorView{},
      static_cast<float>(softmax_scale.value_or(
          1.0 / std::sqrt(static_cast<double>(head_size_)))),
      causal, int{window_size[0]}, int{window_size[1]}, int{0},
      static_cast<float>(softcap), int{0}, int{num_splits},
      OptionalTensorView{}, OptionalTensorView{},
      tvm::ffi::TensorView(dl_out.Get()), tvm::ffi::TensorView(dl_lse.Get()),
      int{1}, int{0}, OptionalTensorView{}, false);

  const auto outputs = result.cast<tvm::ffi::Array<tvm::ffi::Any>>();
  const auto accumulators =
      outputs[0].cast<tvm::ffi::Array<tvm::ffi::Tensor>>();
  const auto actual_num_splits = outputs[1].cast<int>();
  CallCombine(state.runtime_, dl_out, dl_lse, accumulators, max_seqlen_q,
              actual_num_splits);
}

}  // namespace infini::ops
