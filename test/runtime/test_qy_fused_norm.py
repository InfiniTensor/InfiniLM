"""QY fused norm equivalence, strided aliasing, replay, and orderly teardown."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import unittest

import infinicore as ic
import torch


class QyFusedNormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ic.get_device_count('qy') == 0:
            raise unittest.SkipTest('QY device required')
        cls.worker = ThreadPoolExecutor(max_workers=1)
        cls.bf16_mismatches = 0
        cls.bf16_elements = 0
        cls.worker.submit(ic.set_device, 'qy:0').result()

    @classmethod
    def tearDownClass(cls):
        cls.worker.submit(ic.cancel_graph_recording).result()
        cls.worker.submit(ic.sync_device).result()
        cls.worker.submit(ic.set_device, 'cpu').result()
        cls.worker.shutdown(wait=True)
        print(f'BF16 fused versus separate: {cls.bf16_mismatches}/{cls.bf16_elements} elements differ')

    def check_case(self, tokens, dtype, strided, graph):
        torch.manual_seed(42)
        shape = (tokens, 2048)
        a = torch.randn(shape, dtype=dtype)
        b = torch.randn_like(a)
        weight = torch.randn(2048, dtype=dtype)
        width = 2056 if strided else 2048
        xstore = torch.full((tokens, width), -7, dtype=dtype)
        rstore = torch.full_like(xstore, -7)
        def owned(t):
            return ic.from_torch(t).to('qy:0')

        def read(t):
            host = t.contiguous().to('cpu')
            result = torch.empty(tuple(t.shape), dtype=dtype)
            ctypes.memmove(result.data_ptr(), host.data_ptr(), result.numel() * result.element_size())
            return result

        ixstore, irstore, iw = map(owned, (xstore, rstore, weight))
        ix, ir = ixstore.narrow(1, 0, 2048), irstore.narrow(1, 0, 2048)
        graph_obj = None
        if graph:
            ic.start_graph_recording()
            try:
                ic.ops.add_rms_norm(ix, ir, iw, 1e-6, out=ix, residual_out=ir)
                graph_obj = ic.stop_graph_recording()
            except BaseException:
                ic.cancel_graph_recording()
                raise
        for iteration in range(3):
            x, residual = a * (iteration + 1), b
            ix.copy_(owned(x))
            ir.copy_(owned(b))
            rounded = (x.float() + residual.float()).to(dtype)
            ref = rounded.float()
            ref = (ref * torch.rsqrt(ref.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).to(dtype)
            ic.sync_device()
            separate_sum = ic.empty_like(ix)
            separate_out = ic.empty_like(ix)
            ic.ops.add(ix, ir, out=separate_sum)
            ic.ops.rms_norm(separate_sum, iw, 1e-6, out=separate_out)
            if graph_obj is None:
                ic.ops.add_rms_norm(ix, ir, iw, 1e-6, out=ix, residual_out=ir)
            else:
                graph_obj.run()
            ic.sync_device()
            tol = {torch.bfloat16: .016, torch.float16: .002, torch.float32: 2e-5}[dtype]
            torch.testing.assert_close(read(ix), ref, rtol=tol, atol=tol)
            separate = read(separate_out)
            actual = read(ix)
            torch.testing.assert_close(actual, separate, rtol=tol, atol=tol)
            if dtype == torch.bfloat16:
                type(self).bf16_mismatches += (actual != separate).sum().item()
                type(self).bf16_elements += actual.numel()
            torch.testing.assert_close(read(ir), rounded, rtol=0, atol=0)
            if strided:
                self.assertTrue(torch.all(read(ixstore)[:, 2048:] == -7).item())
                self.assertTrue(torch.all(read(irstore)[:, 2048:] == -7).item())

    def test_eager_and_replay(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for tokens in (1, 4, 512):
                for strided in (False, True):
                    for graph in (False, True):
                        with self.subTest(dtype=dtype, tokens=tokens, strided=strided, graph=graph):
                            self.worker.submit(self.check_case, tokens, dtype, strided, graph).result()


if __name__ == '__main__':
    unittest.main()
