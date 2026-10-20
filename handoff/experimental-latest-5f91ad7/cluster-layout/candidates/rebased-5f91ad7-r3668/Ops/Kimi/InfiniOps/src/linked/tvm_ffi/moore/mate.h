#ifndef INFINI_OPS_LINKED_TVM_FFI_MOORE_MATE_H_
#define INFINI_OPS_LINKED_TVM_FFI_MOORE_MATE_H_

#include <ATen/DLConvertor.h>
#include <dlfcn.h>
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace infini::ops::linked::tvm_ffi::moore {

namespace py = pybind11;

using TvmFfiEntry = int(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
using OptionalTensorView = tvm::ffi::Optional<tvm::ffi::TensorView>;

namespace detail {

struct DlLibraryCloser {
  void operator()(void* library) const {
    if (library != nullptr && dlclose(library) != 0) {
      std::fprintf(stderr, "[InfiniOps] MATE library close: %s\n", dlerror());
    }
  }
};

using DlLibrary = std::unique_ptr<void, DlLibraryCloser>;

class DlSymbol {
 public:
  void Load(const std::string& dispatch_name) {
    const auto library_path = FindLoadedLibrary(dispatch_name);
    library_ = DlLibrary{dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL)};
    if (library_ == nullptr) {
      throw std::runtime_error("MATE failed to load " + library_path + ": " +
                               dlerror());
    }

    dlerror();
    entry_ = reinterpret_cast<TvmFfiEntry*>(
        dlsym(library_.get(), ("__tvm_ffi_" + dispatch_name).c_str()));
    if (const auto* error = dlerror(); entry_ == nullptr || error != nullptr) {
      throw std::runtime_error("MATE did not export __tvm_ffi_" +
                               dispatch_name + ": " + error);
    }
  }

  TvmFfiEntry* Entry() const { return entry_; }

 private:
  static std::string FindLoadedLibrary(const std::string& dispatch_name) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    const auto directory = "/" + dispatch_name + "/";
    const auto basename = dispatch_name + ".so";
    while (std::getline(maps, line)) {
      const auto path_pos = line.find(" /");
      if (path_pos == std::string::npos) continue;
      auto path = line.substr(path_pos + 1);
      if (path.find(directory) != std::string::npos ||
          path.substr(path.find_last_of('/') + 1) == basename) {
        return path;
      }
    }
    throw std::runtime_error("MATE did not retain the loaded module for " +
                             dispatch_name);
  }

  DlLibrary library_;
  TvmFfiEntry* entry_{nullptr};
};

class DlPackTensor {
 public:
  explicit DlPackTensor(const at::Tensor& tensor)
      : managed_(at::toDLPack(tensor)) {}

  DlPackTensor(const DlPackTensor&) = delete;
  DlPackTensor& operator=(const DlPackTensor&) = delete;

  ~DlPackTensor() {
    if (managed_ != nullptr && managed_->deleter != nullptr) {
      managed_->deleter(managed_);
    }
  }

  const DLTensor* Get() const { return &managed_->dl_tensor; }

 private:
  DLManagedTensor* managed_;
};

class TvmStreamGuard {
 public:
  TvmStreamGuard(DLDevice device, void* stream) : device_{device} {
    const auto status = TVMFFIEnvSetStream(device.device_type, device.device_id,
                                           stream, &previous_stream_);
    if (status != 0) {
      throw std::runtime_error(
          "MATE failed to select the TVM-FFI MUSA stream (status " +
          std::to_string(status) + ")");
    }
  }

  ~TvmStreamGuard() {
    const auto status = TVMFFIEnvSetStream(
        device_.device_type, device_.device_id, previous_stream_, nullptr);
    if (status != 0) {
      std::fprintf(stderr,
                   "[InfiniOps] MATE failed to restore the TVM-FFI stream "
                   "(status %d)\n",
                   status);
    }
  }

 private:
  DLDevice device_;
  TVMFFIStreamHandle previous_stream_{nullptr};
};

class ModuleRecorder {
 public:
  ModuleRecorder() {
    const auto mate = py::module_::import("mate");
    const auto version = py::str(mate.attr("__version__")).cast<std::string>();
    const auto separator = version.find('+');
    if (version.substr(0, separator) != "0.2.5") {
      throw std::runtime_error(
          "Mate 0.2.5 is required by the Moore native "
          "FlashAttention provider, but found " +
          version);
    }

    forward_ = py::module_::import("mate.jit.attention.fmha.fmha_fwd");
    combine_ = py::module_::import("mate.jit.attention.fmha.fmha_combine");
    original_forward_loader_ = forward_.attr("_fmha_fwd_module");
    original_combine_loader_ = combine_.attr("_fmha_fwd_combine_module");
    forward_names_ = py::list();
    combine_names_ = py::list();

    auto forward_loader = py::cpp_function(
        [forward = forward_, original = original_forward_loader_,
         names = forward_names_](py::object config) mutable {
          names.append(forward.attr("_fmha_fwd_encode")(config));
          return original(config);
        });
    auto combine_loader = py::cpp_function(
        [combine = combine_, original = original_combine_loader_,
         names = combine_names_](py::object config) mutable {
          names.append(combine.attr("_fmha_fwd_combine_encode")(config));
          return original(config);
        });
    forward_.attr("_fmha_fwd_module") = forward_loader;
    combine_.attr("_fmha_fwd_combine_module") = combine_loader;
  }

  ModuleRecorder(const ModuleRecorder&) = delete;
  ModuleRecorder& operator=(const ModuleRecorder&) = delete;

  ~ModuleRecorder() {
    if (forward_) forward_.attr("_fmha_fwd_module") = original_forward_loader_;
    if (combine_) {
      combine_.attr("_fmha_fwd_combine_module") = original_combine_loader_;
    }
  }

  py::module_ Forward() const { return forward_; }

  std::string ForwardName() const { return Name(forward_names_); }

  std::optional<std::string> CombineName() const {
    if (py::len(combine_names_) == 0) return std::nullopt;
    return Name(combine_names_);
  }

 private:
  static std::string Name(const py::list& names) {
    if (py::len(names) != 1) {
      throw std::runtime_error(
          "MATE selected an unexpected number of FlashAttention modules");
    }
    return py::str(names[0]).cast<std::string>();
  }

  py::module_ forward_;
  py::module_ combine_;
  py::object original_forward_loader_;
  py::object original_combine_loader_;
  py::list forward_names_;
  py::list combine_names_;
};

}  // namespace detail

class MateFmhaRuntime {
 public:
  void Load(const std::string& forward_name,
            const std::optional<std::string>& combine_name) {
    forward_.Load(forward_name);
    if (combine_name.has_value()) combine_.Load(*combine_name);
  }

  detail::DlSymbol& Forward() { return forward_; }

  detail::DlSymbol* Combine() {
    return combine_.Entry() == nullptr ? nullptr : &combine_;
  }

 private:
  detail::DlSymbol forward_;
  detail::DlSymbol combine_;
};

}  // namespace infini::ops::linked::tvm_ffi::moore

#endif  // INFINI_OPS_LINKED_TVM_FFI_MOORE_MATE_H_
