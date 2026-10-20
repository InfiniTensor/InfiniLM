#include "linked/tvm_ffi/moore/ops/flash_attn_varlen_func/mate.h"

#include <tvm/ffi/container/array.h>

#include <cmath>
#include <stdexcept>
#include <vector>

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
                 mate::detail::DlPackTensor& cu_seqlens_q, int max_seqlen_q,
                 int num_splits) {
  auto* combine = runtime.Combine();
  if (num_splits > 1 && combine == nullptr) {
    throw std::runtime_error(
        "MATE selected split-KV execution without a combine kernel");
  }
  if (num_splits <= 1) return;

  tvm::ffi::Function::InvokeExternC(
      nullptr, combine->Entry(),
      OptionalTensorView{tvm::ffi::TensorView(cu_seqlens_q.Get())},
      OptionalTensorView{}, tvm::ffi::Optional<int>{max_seqlen_q},
      tvm::ffi::TensorView(out.Get()), tvm::ffi::TensorView(lse.Get()),
      tvm::ffi::TensorView(accumulators[0]),
      tvm::ffi::TensorView(accumulators[1]), OptionalTensorView{},
      int{num_splits});
}

}  // namespace

struct Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 16>::MateState {
  linked::tvm_ffi::moore::MateFmhaRuntime runtime_;
  std::optional<at::Tensor> paged_seqused_k_;
  std::optional<at::Tensor> internal_lse_;
};

Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 16>::~Operator() = default;

void Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 16>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  std::lock_guard lock{runtime_mutex_};
  Call(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,
       max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
       window_size, softcap, deterministic, return_attn_probs, out, softmax_lse,
       s_dmask);
}

void Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 16>::Call(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  (void)s_dmask;
  TORCH_CHECK(!alibi_slopes.has_value(), "MATE ALiBi is not supported");
  TORCH_CHECK(dropout_p == 0.0, "MATE attention dropout is not supported");
  TORCH_CHECK(!deterministic, "MATE deterministic attention is not supported");
  TORCH_CHECK(!return_attn_probs,
              "MATE attention probabilities are not supported");

  if (!state_) state_ = std::make_shared<MateState>();
  auto& state = *state_;

  const typename C10<Device::Type::kMoore>::StreamGuard stream_guard{
      C10<Device::Type::kMoore>::GetStreamFromExternal(stream_, device_index_)};

  auto at_q =
      ToAtenTensor<Device::Type::kMoore>(const_cast<void*>(q.data()), q_shape_,
                                         q_strides_, q_dtype_, device_index_);
  auto at_k =
      ToAtenTensor<Device::Type::kMoore>(const_cast<void*>(k.data()), k_shape_,
                                         k_strides_, k_dtype_, device_index_);
  auto at_v =
      ToAtenTensor<Device::Type::kMoore>(const_cast<void*>(v.data()), v_shape_,
                                         v_strides_, v_dtype_, device_index_);
  auto at_cu_seqlens_q = ToAtenTensor<Device::Type::kMoore>(
      const_cast<void*>(cu_seqlens_q.data()), cu_seqlens_q_shape_,
      cu_seqlens_q_strides_, cu_seqlens_q_dtype_, device_index_);
  auto at_cu_seqlens_k = ToAtenTensor<Device::Type::kMoore>(
      const_cast<void*>(cu_seqlens_k.data()), cu_seqlens_k_shape_,
      cu_seqlens_k_strides_, cu_seqlens_k_dtype_, device_index_);
  auto at_out = ToAtenTensor<Device::Type::kMoore>(
      out.data(), out_shape_, out_strides_, out_dtype_, device_index_);

  std::optional<at::Tensor> at_block_table;
  if (block_table.has_value()) {
    at_block_table.emplace(ToAtenTensor<Device::Type::kMoore>(
        const_cast<void*>(block_table->data()), block_table_shape_,
        block_table_strides_, block_table_dtype_, device_index_));
  }

  if (block_table.has_value() && !state.paged_seqused_k_.has_value()) {
    state.paged_seqused_k_.emplace(
        (at_cu_seqlens_k.slice(0, 1) - at_cu_seqlens_k.slice(0, 0, -1))
            .contiguous());
  }
  at::Tensor at_lse;
  if (softmax_lse.has_value()) {
    at_lse = ToAtenTensor<Device::Type::kMoore>(
        const_cast<void*>(softmax_lse->data()), softmax_lse_shape_,
        softmax_lse_strides_, softmax_lse_dtype_, device_index_);
  } else {
    if (!state.internal_lse_.has_value()) {
      state.internal_lse_.emplace(
          at::empty({static_cast<int64_t>(q_shape_[1]),
                     static_cast<int64_t>(q_shape_[0])},
                    at_out.options().dtype(at::kFloat)));
    }
    at_lse = *state.internal_lse_;
  }

  if (state.runtime_.Forward().Entry() == nullptr) {
    py::gil_scoped_acquire gil;
    try {
      mate::detail::ModuleRecorder recorder;
      recorder.Forward().attr("_fmha_fwd")(
          py::arg("q") = at_q, py::arg("k") = at_k, py::arg("v") = at_v,
          py::arg("k_new") = py::none(), py::arg("v_new") = py::none(),
          py::arg("q_v") = py::none(),
          py::arg("cu_seqlens_q") = at_cu_seqlens_q,
          py::arg("cu_seqlens_k") =
              (block_table.has_value() ? py::none()
                                       : py::cast(at_cu_seqlens_k)),
          py::arg("cu_seqlens_k_new") = py::none(),
          py::arg("seqused_q") = py::none(),
          py::arg("seqused_k") = PythonTensor(state.paged_seqused_k_),
          py::arg("max_seqlen_q") = max_seqlen_q,
          py::arg("max_seqlen_k") = max_seqlen_k,
          py::arg("page_table") = PythonTensor(at_block_table),
          py::arg("kv_batch_idx") = py::none(),
          py::arg("leftpad_k") = py::none(), py::arg("rotary_cos") = py::none(),
          py::arg("rotary_sin") = py::none(),
          py::arg("seqlens_rotary") = py::none(),
          py::arg("q_descale") = py::none(), py::arg("k_descale") = py::none(),
          py::arg("v_descale") = py::none(),
          py::arg("softmax_scale") = static_cast<float>(softmax_scale.value_or(
              1.0 / std::sqrt(static_cast<double>(q_shape_[2])))),
          py::arg("is_causal") = causal,
          py::arg("window_size_left") = window_size[0],
          py::arg("window_size_right") = window_size[1],
          py::arg("attention_chunk") = 0,
          py::arg("learnable_sink") = py::none(),
          py::arg("softcap") = static_cast<float>(softcap),
          py::arg("is_rotary_interleaved") = false,
          py::arg("scheduler_metadata") = py::none(), py::arg("num_splits") = 0,
          py::arg("pack_gqa") =
              (q_shape_[1] !=
               (block_table.has_value() ? k_shape_[3] : k_shape_[2])),
          py::arg("mp_margin") = 0, py::arg("return_lse") = true,
          py::arg("lse") = at_lse, py::arg("out") = at_out,
          py::arg("cp_world_size") = 1, py::arg("cp_rank") = 0,
          py::arg("cp_tot_seqused_k") = py::none(), py::arg("only_qv") = false);
      state.runtime_.Load(recorder.ForwardName(), recorder.CombineName());
    } catch (const py::error_already_set& error) {
      TORCH_CHECK(false, "MATE flash_attn_varlen_func bootstrap failed: ",
                  error.what());
    }
    return;
  }

  mate::detail::DlPackTensor dl_q{at_q};
  mate::detail::DlPackTensor dl_k{at_k};
  mate::detail::DlPackTensor dl_v{at_v};
  mate::detail::DlPackTensor dl_cu_q{at_cu_seqlens_q};
  mate::detail::DlPackTensor dl_cu_k{at_cu_seqlens_k};
  mate::detail::DlPackTensor dl_out{at_out};
  mate::detail::DlPackTensor dl_lse{at_lse};
  std::optional<mate::detail::DlPackTensor> dl_block_table;
  std::optional<mate::detail::DlPackTensor> dl_seqused_k;
  if (at_block_table.has_value()) {
    dl_block_table.emplace(*at_block_table);
    dl_seqused_k.emplace(*state.paged_seqused_k_);
  }

  const mate::detail::TvmStreamGuard tvm_stream_guard{
      dl_q.Get()->device,
      C10<Device::Type::kMoore>::GetStreamFromExternal(stream_, device_index_)};

  auto result = tvm::ffi::Function::InvokeExternC(
      nullptr, state.runtime_.Forward().Entry(),
      tvm::ffi::TensorView(dl_q.Get()), tvm::ffi::TensorView(dl_k.Get()),
      tvm::ffi::TensorView(dl_v.Get()), OptionalTensorView{},
      OptionalTensorView{}, OptionalTensorView{},
      tvm::ffi::TensorView(dl_cu_q.Get()),
      block_table.has_value()
          ? OptionalTensorView{}
          : OptionalTensorView{tvm::ffi::TensorView(dl_cu_k.Get())},
      OptionalTensorView{}, OptionalTensorView{},
      dl_seqused_k.has_value()
          ? OptionalTensorView{tvm::ffi::TensorView(dl_seqused_k->Get())}
          : OptionalTensorView{},
      tvm::ffi::Optional<int>{static_cast<int>(max_seqlen_q)},
      tvm::ffi::Optional<int>{static_cast<int>(max_seqlen_k)},
      dl_block_table.has_value()
          ? OptionalTensorView{tvm::ffi::TensorView(dl_block_table->Get())}
          : OptionalTensorView{},
      OptionalTensorView{}, OptionalTensorView{}, OptionalTensorView{},
      OptionalTensorView{}, OptionalTensorView{}, OptionalTensorView{},
      OptionalTensorView{}, OptionalTensorView{}, OptionalTensorView{},
      static_cast<float>(softmax_scale.value_or(
          1.0 / std::sqrt(static_cast<double>(q_shape_[2])))),
      causal, int{window_size[0]}, int{window_size[1]}, int{0},
      static_cast<float>(softcap), int{0}, int{0}, OptionalTensorView{},
      OptionalTensorView{}, tvm::ffi::TensorView(dl_out.Get()),
      tvm::ffi::TensorView(dl_lse.Get()), int{1}, int{0}, OptionalTensorView{},
      false);

  const auto outputs = result.cast<tvm::ffi::Array<tvm::ffi::Any>>();
  const auto accumulators =
      outputs[0].cast<tvm::ffi::Array<tvm::ffi::Tensor>>();
  const auto num_splits = outputs[1].cast<int>();
  CallCombine(state.runtime_, dl_out, dl_lse, accumulators, dl_cu_q,
              static_cast<int>(max_seqlen_q), num_splits);
}

}  // namespace infini::ops
