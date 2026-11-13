"""Greedy sampling contract; run with INFINILM_TEST_DEVICE=qy on QY hosts."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import infinicore
import torch
from infinicore import ops


class GreedySampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = os.environ.get("INFINILM_TEST_DEVICE", "cuda")
        cls.device = f"{cls.backend}:0"
        # Match RankWorker: device TLS is destroyed before process teardown.
        cls.worker = ThreadPoolExecutor(max_workers=1)
        cls.addClassCleanup(cls.worker.shutdown, wait=True)
        cls.worker.submit(cls._init_device).result()
        cls.addClassCleanup(cls._release_device)

    @classmethod
    def _init_device(cls):
        if infinicore.get_device_count(cls.backend) == 0:
            raise unittest.SkipTest(f"{cls.backend} device is required")
        infinicore.set_device(cls.device)

    @classmethod
    def _release_device(cls):
        cls.worker.submit(infinicore.set_device, "cpu").result()

    def check_sample(self, rows, vocab, dtype, scalar=False):
        self.worker.submit(self._check_sample, rows, vocab, dtype, scalar).result()

    def _check_sample(self, rows, vocab, dtype, scalar):
        logits = torch.zeros((rows, vocab), dtype=dtype)
        expected = []
        for row in range(rows):
            index = (3, vocab // 2, 17, vocab - 1)[row]
            logits[row, index] = row + 1
            expected.append(index)
        if scalar:
            logits = logits.view(vocab)
        host_logits = infinicore.from_torch(logits)
        scores = host_logits.to(self.device)
        # Guard values also detect writes beyond a narrowed output view.
        sentinels = torch.full((rows + 2,), -777, dtype=torch.int64)
        host_output = infinicore.from_torch(sentinels)
        output = host_output.to(self.device)
        infinicore.sync_stream()
        indices = output.narrow(0, 1, rows)
        if scalar:
            indices = indices.view([])
        for _ in range(2):
            ops.random_sample(scores, 0.5, 1.0, 1, 0.0, out=indices)
            infinicore.sync_stream()
            self.assertEqual(output.to_numpy().tolist(), [-777, *expected, -777])

    def test_batched_greedy(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows in (1, 2, 4):
                for vocab in (97, 130560):
                    with self.subTest(dtype=dtype, rows=rows, vocab=vocab):
                        self.check_sample(rows, vocab, dtype)

    def test_scalar_greedy(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check_sample(1, 97, dtype, scalar=True)


if __name__ == "__main__":
    unittest.main()
