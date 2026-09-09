import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class CoreInitializerSyncTest(unittest.TestCase):
    def test_ones_is_graph_aware_and_uses_modern_fill(self) -> None:
        header = read_source("csrc/infinicore/include/infinicore/ops/ones.hpp")
        source = read_source("csrc/infinicore/src/ops/ones/ones.cc")
        backend = read_source("csrc/infinicore/src/ops/ones/ones_infiniops.cc")

        self.assertIn("INFINICORE_GRAPH_OP_CLASS(Ones, Tensor);", header)
        self.assertIn("INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Ones);", source)
        self.assertIn("INFINICORE_GRAPH_OP_RECORD_OR_RUN(Ones, output);", source)
        self.assertIn("graph::GraphTensor output_tensor;", backend)
        self.assertIn("handle.set_stream(context::getStream())", backend)
        self.assertIn("infini::ops::Fill::Call(", backend)
        for legacy_api in (
            "infiniopCreateOnesDescriptor",
            "infiniopOnesDescriptor_t",
            "infiniopGetOnesWorkspaceSize",
            "infiniopOnes(",
            "#include <infiniop.h>",
        ):
            self.assertNotIn(legacy_api, backend)

    def test_cpu_ones_writes_numeric_one_for_every_migrated_dtype(self) -> None:
        source = read_source("csrc/infinicore/src/ops/ones/ones.cc")

        for dtype in (
            "kInt8",
            "kInt16",
            "kInt32",
            "kInt64",
            "kUInt8",
            "kUInt16",
            "kUInt32",
            "kUInt64",
            "kFloat16",
            "kBFloat16",
            "kFloat32",
            "kFloat64",
        ):
            self.assertIn(f"DataType::{dtype}", source)
        self.assertIn("utils::cast<fp16_t, float>(1.0f)", source)
        self.assertIn("utils::cast<bf16_t, float>(1.0f)", source)
        self.assertNotIn("setDeviceMemory", source)

    def test_zeros_is_a_graph_recorded_infini_rt_memset(self) -> None:
        header = read_source("csrc/infinicore/include/infinicore/ops/zeros.hpp")
        source = read_source("csrc/infinicore/src/ops/zeros/zeros.cc")
        backend = read_source("csrc/infinicore/src/ops/zeros/zeros_infinirt.cc")

        self.assertIn("INFINICORE_GRAPH_OP_CLASS(Zeros, Tensor);", header)
        self.assertIn("INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Zeros);", source)
        self.assertIn("INFINICORE_GRAPH_OP_RECORD_OR_RUN(Zeros, output);", source)
        self.assertIn("graph::GraphTensor output;", backend)
        self.assertIn("context::setDeviceMemoryAsync(", backend)
        self.assertIn("context::getStream()", backend)
        self.assertIn("INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Zeros", backend)

    def test_tensor_factories_dispatch_initializers_and_skip_empty_tensors(
        self,
    ) -> None:
        source = read_source("csrc/infinicore/src/tensor/tensor.cc")
        zeros_start = source.index("std::shared_ptr<TensorImpl> TensorImpl::zeros")
        ones_start = source.index("std::shared_ptr<TensorImpl> TensorImpl::ones")
        from_blob_start = source.index(
            "std::shared_ptr<TensorImpl> TensorImpl::from_blob"
        )
        zeros_body = source[zeros_start:ones_start]
        ones_body = source[ones_start:from_blob_start]

        self.assertIn("if (result->nbytes() != 0)", zeros_body)
        self.assertIn("op::zeros_(Tensor{result})", zeros_body)
        self.assertNotIn("setDeviceMemoryAsync", zeros_body)
        self.assertIn("if (result->nbytes() != 0)", ones_body)
        self.assertIn("op::ones_(Tensor{result})", ones_body)
        self.assertNotIn("TODO", ones_body)

        ops_header = read_source("csrc/infinicore/include/infinicore/ops.hpp")
        self.assertIn('#include "ops/ones.hpp"', ops_header)
        self.assertIn('#include "ops/zeros.hpp"', ops_header)


if __name__ == "__main__":
    unittest.main()
