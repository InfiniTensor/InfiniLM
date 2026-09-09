import ctypes
import unittest

import infinicore
import torch
import torch.nn.functional as torch_functional
from infinicore import ops


def to_torch_bfloat16(tensor):
    source = tensor.to("cpu").contiguous()
    result = torch.empty(source.shape, dtype=torch.bfloat16)
    ctypes.memmove(result.data_ptr(), source.data_ptr(), source.numel() * 2)
    return result


class PackedSwiGLUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if infinicore.get_device_count("cuda") == 0:
            raise unittest.SkipTest("NVIDIA device is required")

    def setUp(self):
        infinicore.set_device("cuda:0")
        gate = torch.tensor(
            [[[-3.0, -0.5, 0.0, 2.0], [0.25, 1.0, 3.0, -2.0]]],
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        up = torch.tensor(
            [[[0.5, -2.0, 4.0, 1.5], [-3.0, 0.75, -0.5, 2.0]]],
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        packed = torch.cat((gate, up), dim=-1)
        self.input = infinicore.from_torch(packed)
        self.expected = (
            (torch_functional.silu(gate.float()) * up.float()).to(torch.bfloat16).cpu()
        )

    def assert_matches_reference(self, output):
        infinicore.sync_stream()
        actual = to_torch_bfloat16(output)
        torch.testing.assert_close(
            actual.float(), self.expected.float(), rtol=2e-2, atol=2e-2
        )

    def test_eager_packed_gate_up_matches_torch(self):
        output = ops.silu_and_mul(self.input)

        self.assertEqual(output.shape, [1, 2, 4])
        self.assert_matches_reference(output)

    def test_direct_packed_path_matches_legacy_repack(self):
        gate = self.input.narrow(2, 0, 4)
        up = self.input.narrow(2, 4, 4)
        legacy_packed = infinicore.empty(
            self.input.shape, dtype=infinicore.bfloat16, device="cuda:0"
        )
        legacy_packed.narrow(2, 0, 4).copy_(gate)
        legacy_packed.narrow(2, 4, 4).copy_(up)

        direct = ops.silu_and_mul(self.input)
        legacy = ops.silu_and_mul(legacy_packed)
        infinicore.sync_stream()

        self.assertTrue(
            torch.equal(to_torch_bfloat16(direct), to_torch_bfloat16(legacy))
        )
        self.assert_matches_reference(direct)

    def test_graph_packed_gate_up_matches_torch(self):
        zero = infinicore.zeros(
            self.input.shape, dtype=infinicore.bfloat16, device="cuda:0"
        )
        infinicore.sync_stream()
        infinicore.start_graph_recording()
        try:
            packed_intermediate = ops.add(self.input, zero)
            output = ops.silu_and_mul(packed_intermediate)
            graph = infinicore.stop_graph_recording()
        except BaseException:
            if infinicore.is_graph_recording():
                infinicore.cancel_graph_recording()
            raise

        del packed_intermediate
        churn = [
            infinicore.empty(
                self.input.shape, dtype=infinicore.bfloat16, device="cuda:0"
            )
            for _ in range(4)
        ]
        del churn

        for _ in range(2):
            graph.run()
            self.assert_matches_reference(output)


if __name__ == "__main__":
    unittest.main()
